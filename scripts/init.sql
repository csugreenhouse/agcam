-- =========================
-- 0) (Optional) Database
-- =========================
-- NOTE: CREATE DATABASE cannot run inside a transaction block and is usually run separately.
-- CREATE DATABASE agcam;

-- After creating it, connect:
-- \c agcam


-- =========================
-- 1) Drop existing objects
-- =========================
-- see if table exists before dropping to avoid errors when running this script multiple times
-- DROP TABLE IF EXISTS height_view CASCADE;


-- =========================
-- 2) Types
-- =========================
-- CREATE TYPE view_type_enum AS ENUM ('height', 'width');


-- =========================
-- 3) Tables
-- =========================
CREATE TABLE imgIndex (
  cam_id INTEGER NOT NULL,
  file_path TEXT NOT NULL UNIQUE
);

-- =========================
-- 3.5) Functions and Triggers
-- =========================


-- =========================
-- 4) Seed data
-- =========================