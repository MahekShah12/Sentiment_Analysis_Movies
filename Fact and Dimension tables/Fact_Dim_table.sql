use DATAWAREHOUSE
DROP TABLE IF EXISTS dbo.Fact_MoviePerformance;
DROP TABLE IF EXISTS dbo.Dim_Actor;
DROP TABLE IF EXISTS dbo.Dim_Director;
DROP TABLE IF EXISTS dbo.Dim_Movie;
DROP TABLE IF EXISTS dbo.Dim_Netflix;
DROP TABLE IF EXISTS dbo.Dim_BoxOffice;
DROP TABLE IF EXISTS dbo.Dim_TMDb;
GO




-- ACTOR
CREATE TABLE dbo.Dim_Actor (
    actor_sk INT IDENTITY(1,1) PRIMARY KEY,
    actor_id VARCHAR(50),
    name NVARCHAR(150),
    birthYear VARCHAR(50)
);
GO

-- DIRECTOR
CREATE TABLE dbo.Dim_Director (
    director_sk INT IDENTITY(1,1) PRIMARY KEY,
    movie_id VARCHAR(50),
    imdb_id VARCHAR(50),
    title NVARCHAR(255),
    country NVARCHAR(100),
    director NVARCHAR(150),
    genre NVARCHAR(100),
    festival NVARCHAR(100)
);
GO

-- MOVIE
CREATE TABLE dbo.Dim_Movie (
    movie_sk INT IDENTITY(1,1) PRIMARY KEY,
    movie_id VARCHAR(50),
    title NVARCHAR(255),
    genre NVARCHAR(100),
    runtime NVARCHAR(50),
    year VARCHAR(50)
);
GO

-- NETFLIX
CREATE TABLE dbo.Dim_Netflix (
    netflix_sk INT IDENTITY(1,1) PRIMARY KEY,
    show_id VARCHAR(50),
    type NVARCHAR(50),
    title NVARCHAR(255),
    director NVARCHAR(150),
    cast NVARCHAR(500),
    country NVARCHAR(150),
    release_year VARCHAR(50),
    duration NVARCHAR(50),
    listed_in NVARCHAR(255)
);
GO

-- BOX OFFICE
CREATE TABLE dbo.Dim_BoxOffice (
    boxoffice_sk INT IDENTITY(1,1) PRIMARY KEY,
    genre NVARCHAR(100),
    rank NVARCHAR(50),
    release_title NVARCHAR(255),
    worldwide_gross NVARCHAR(50),
    domestic_percent NVARCHAR(50),
    foreign_percent NVARCHAR(50),
    year NVARCHAR(50),
    rating NVARCHAR(50),
    production_countries NVARCHAR(150)
);
GO

--TMDB REVIEWS
CREATE TABLE dbo.Dim_TMDb (
    tmdb_sk INT IDENTITY(1,1) PRIMARY KEY,
    movie_id VARCHAR(50),
    title NVARCHAR(255),
    author NVARCHAR(100),
    review NVARCHAR(3000),
    rating NVARCHAR(50),
    sentiment_score NVARCHAR(50),
    sentiment_score_numeric FLOAT NULL,
    sentiment_label NVARCHAR(50) NULL
);
GO




---sample NOT TO RUN--
CREATE TABLE dbo.Fact_MoviePerformance (
    fact_sk INT IDENTITY(1,1) PRIMARY KEY,
    movie_sk INT NULL,
    director_sk INT NULL,
    netflix_sk INT NULL,
    boxoffice_sk INT NULL,
    tmdb_sk INT NULL,
    actor_sk INT NULL,
    total_reviews INT NULL,
    avg_rating FLOAT NULL,
    sentiment_avg FLOAT NULL
);
GO


-- Dim_Movie
INSERT INTO dbo.Dim_Movie (movie_id, title, genre, runtime, year)
SELECT DISTINCT
    movie_id, title, genre, runtime, year
FROM dbo.Movies
WHERE title IS NOT NULL;
GO

-- Dim_Actor
INSERT INTO dbo.Dim_Actor (actor_id, name, birthYear)
SELECT DISTINCT
    actor_id, name, birthYear
FROM dbo.Actor
WHERE name IS NOT NULL;
GO

--Dim_Director
INSERT INTO dbo.Dim_Director (movie_id, imdb_id, title, country, director, genre, festival)
SELECT DISTINCT
    movie_id, imdb_id, title, country, director, genre, festival
FROM dbo.Director
WHERE director IS NOT NULL;
GO

--Dim_Netflix
INSERT INTO dbo.Dim_Netflix (show_id, type, title, director, cast, country, release_year, duration, listed_in)
SELECT DISTINCT
    show_id, type, title, director, cast, country, release_year, duration, listed_in
FROM dbo.Netflix
WHERE title IS NOT NULL;
GO

-- Dim_BoxOffice
INSERT INTO dbo.Dim_BoxOffice (genre, rank, release_title, worldwide_gross, domestic_percent, foreign_percent, year, rating, production_countries)
SELECT DISTINCT
    genre, rank, release_title, worldwide_gross, domestic_percent, foreign_percent, year, rating, production_countries
FROM dbo.BoxOffice
WHERE release_title IS NOT NULL;
GO

-- Dim_TMDb
INSERT INTO dbo.Dim_TMDb (
    movie_id,
    title,
    author,
    review,
    rating,
    sentiment_score,
    sentiment_score_numeric
)
SELECT
    movie_id,
    title,
    author,
    review,
    TRY_CAST(rating AS FLOAT) AS rating,
    sentiment_score,

    CASE 
        WHEN LOWER(LTRIM(RTRIM(sentiment_score))) = 'positive' THEN 1
        WHEN LOWER(LTRIM(RTRIM(sentiment_score))) = 'negative' THEN -1
        WHEN LOWER(LTRIM(RTRIM(sentiment_score))) = 'neutral' THEN 0
        ELSE NULL
    END AS sentiment_score_numeric
FROM dbo.TMdB
WHERE title IS NOT NULL;
GO



SELECT TOP 10 * FROM dbo.Dim_Actor;
SELECT TOP 10 * FROM dbo.Dim_Director;
SELECT TOP 10 * FROM dbo.Dim_BoxOffice;
SELECT TOP 10 * FROM dbo.Dim_Movie;
SELECT top 10 * FROM dbo.Dim_Netflix;
SELECT top 10 * FROM dbo.Dim_TMDb;
GO

---FACT TABLE FINAL AS OF NOW--
---------------Table1 : working well
USE DATAWAREHOUSE
GO

-- Drop existing fact table
DROP TABLE IF EXISTS dbo.Fact_MoviePerformance;
GO

-- Create Fact Table
CREATE TABLE dbo.Fact_MoviePerformance (
    fact_sk INT IDENTITY(1,1) PRIMARY KEY,
    movie_sk INT NOT NULL,
    director_sk INT NOT NULL,
    netflix_sk INT NULL,
    boxoffice_sk INT NULL,
    tmdb_sk INT NULL,
    actor_sk INT NULL,
    total_reviews INT NULL,
    avg_rating FLOAT NULL,
    sentiment_avg FLOAT NULL
);
GO

-- Insert data into Fact Table
INSERT INTO dbo.Fact_MoviePerformance (
    movie_sk, director_sk, netflix_sk, boxoffice_sk, tmdb_sk, actor_sk,
    total_reviews, avg_rating, sentiment_avg
)
SELECT 
    m.movie_sk,
    d.director_sk,
    n.netflix_sk,
    b.boxoffice_sk,
    MAX(t.tmdb_sk) AS tmdb_sk,  -- one sample TMDb entry per movie
    a.actor_sk,
    COUNT(t.tmdb_sk) AS total_reviews,
    AVG(t.rating) AS avg_rating,
    AVG(t.sentiment_score_numeric) AS sentiment_avg
FROM dbo.Dim_Movie m
INNER JOIN dbo.Dim_Director d 
    ON LOWER(LTRIM(RTRIM(m.title))) = LOWER(LTRIM(RTRIM(d.title)))
LEFT JOIN dbo.Dim_Netflix n 
    ON LOWER(LTRIM(RTRIM(m.title))) = LOWER(LTRIM(RTRIM(n.title)))
LEFT JOIN dbo.Dim_BoxOffice b 
    ON LOWER(LTRIM(RTRIM(m.title))) = LOWER(LTRIM(RTRIM(b.release_title)))
LEFT JOIN dbo.Dim_TMDb t 
    ON LOWER(LTRIM(RTRIM(m.title))) = LOWER(LTRIM(RTRIM(t.title)))
LEFT JOIN dbo.Dim_Actor a 
    ON a.actor_id IN (
        SELECT TOP 1 actor_id FROM dbo.Dim_Actor ORDER BY NEWID()
    )
WHERE d.director_sk IS NOT NULL
  AND t.tmdb_sk IS NOT NULL
GROUP BY 
    m.movie_sk, d.director_sk, n.netflix_sk, b.boxoffice_sk, a.actor_sk;
GO

-- Check sample output
SELECT  *
FROM dbo.Fact_MoviePerformance
ORDER BY fact_sk;
GO
