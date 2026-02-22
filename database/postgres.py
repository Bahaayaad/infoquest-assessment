
import logging
import psycopg2
import psycopg2.extras
from config import settings
from models.candidate import CandidateProfile

logger = logging.getLogger(__name__)


def get_connection():
    try:
        conn = psycopg2.connect(settings.postgres_url)
        return conn
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL: %s", e)
        raise


def fetch_all_candidates() -> list[CandidateProfile]:
    """
    Pull every candidate with their skills, education, languages,
    and work history all joined together.
    """
    logger.info("Fetching all candidates from PostgreSQL")

    try:
        conn = get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except Exception as e:
        logger.error("Could not open DB cursor: %s", e)
        raise

    try:
        #I used a JOIN query to pull everything needed into one flat row per candidate. since the DB is fully normalized
        cur.execute("""
                    WITH
                        -- Aggregate all skills per candidate in one pass
                        all_skills AS (SELECT cs.candidate_id,
                                              string_agg(DISTINCT s.name, ', ' ORDER BY s.name) AS skills,
                                              string_agg(DISTINCT s.name, ', ' ORDER BY s.name)
                                                                                                   FILTER (WHERE cs.proficiency_level = 'Expert') AS top_skills
                                       FROM candidate_skills cs
                                                JOIN skills s ON s.id = cs.skill_id
                                       GROUP BY cs.candidate_id),

                        -- Aggregate work history per candidate in one pass
                        work_hist AS (SELECT we.candidate_id,
                                             string_agg(
                                                     we.job_title || ' at ' || comp.name,
                                                     ' | ' ORDER BY we.start_date DESC
                                             ) AS work_history
                                      FROM work_experience we
                                               JOIN companies comp ON comp.id = we.company_id
                                      GROUP BY we.candidate_id),

                        -- Aggregate education per candidate in one pass
                        edu_agg AS (SELECT e.candidate_id,
                                           string_agg(
                                                   d.name || ' in ' || fos.name || ' at ' || inst.name,
                                                   ' | ' ORDER BY e.graduation_year DESC NULLS LAST
                                           ) AS education
                                    FROM education e
                                             JOIN degrees d ON d.id = e.degree_id
                                             JOIN fields_of_study fos ON fos.id = e.field_of_study_id
                                             JOIN institutions inst ON inst.id = e.institution_id
                                    GROUP BY e.candidate_id),

                        -- Aggregate languages per candidate in one pass
                        lang_agg AS (SELECT cl.candidate_id,
                                            string_agg(
                                                    l.name || ' (' || pl.name || ')',
                                                    ', ' ORDER BY pl.rank DESC
                                            ) AS languages
                                     FROM candidate_languages cl
                                              JOIN languages l ON l.id = cl.language_id
                                              JOIN proficiency_levels pl ON pl.id = cl.proficiency_level_id
                                     GROUP BY cl.candidate_id),

                        -- Get only the most recent current job per candidate
                        current_job AS (SELECT DISTINCT
                    ON (we.candidate_id)
                        we.candidate_id,
                        we.job_title,
                        we.description,
                        comp.name AS company_name,
                        comp.industry
                    FROM work_experience we
                        JOIN companies comp
                    ON comp.id = we.company_id
                    WHERE we.is_current = true
                    ORDER BY we.candidate_id, we.start_date DESC
                        )

                    -- Final SELECT: simple JOINs against pre-aggregated CTEs
                    SELECT c.id::text, c.first_name || ' ' || c.last_name AS name,
                           c.headline,
                           c.email,
                           c.years_of_experience,
                           ci.name         AS city,
                           co.name         AS country,
                           cj.job_title    AS current_title,
                           cj.company_name AS current_company,
                           cj.industry,
                           cj.description  AS job_description,
                           wh.work_history,
                           sk.skills,
                           sk.top_skills,
                           ed.education,
                           la.languages
                    FROM candidates c
                             LEFT JOIN cities ci ON ci.id = c.city_id
                             LEFT JOIN countries co ON co.id = ci.country_id
                             LEFT JOIN current_job cj ON cj.candidate_id = c.id
                             LEFT JOIN all_skills sk ON sk.candidate_id = c.id
                             LEFT JOIN work_hist wh ON wh.candidate_id = c.id
                             LEFT JOIN edu_agg ed ON ed.candidate_id = c.id
                             LEFT JOIN lang_agg la ON la.candidate_id = c.id
                    ORDER BY c.created_at DESC
                    """)
        rows = cur.fetchall()
    except Exception as e:
        logger.error("SQL query failed: %s", e)
        raise
    finally:
        cur.close()
        conn.close()

    logger.info("SQL returned %d rows", len(rows))

    candidates = []
    skipped = 0
    for row in rows:
        try:
            candidates.append(CandidateProfile(**dict(row)))
        except Exception as e:
            skipped += 1
            logger.warning("Skipping malformed row id=%s: %s", row.get("id"), e)

    if skipped:
        logger.warning("Skipped %d malformed rows during fetch", skipped)

    logger.info("Returning %d valid candidates", len(candidates))
    return candidates


def count_candidates() -> int:
    logger.debug("Counting candidates in DB")
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM candidates")
        result = cur.fetchone()[0]
        cur.close()
        conn.close()
        logger.debug("Total candidates in DB: %d", result)
        return result
    except Exception as e:
        logger.error("Failed to count candidates: %s", e)
        raise
