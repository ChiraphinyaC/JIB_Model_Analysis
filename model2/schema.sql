-- 1. ส่วนสร้างโครงสร้าง Data Warehouse (Star Schema DDL)
CREATE TABLE IF NOT EXISTS Dim_Campaign_Type (
    Type_ID INTEGER PRIMARY KEY,
    Campaign_Type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Target_Audience (
    Audience_ID INTEGER PRIMARY KEY,
    Target_Audience TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Channel (
    Channel_ID INTEGER PRIMARY KEY,
    Channel_Name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS Dim_Date (
    Date_Key INTEGER PRIMARY KEY,
    Start_Date TEXT,
    End_Date TEXT,
    Start_Month INTEGER,
    Start_Quarter INTEGER
);

CREATE TABLE IF NOT EXISTS Fact_Marketing_Campaign (
    Campaign_ID TEXT PRIMARY KEY,
    Type_ID INTEGER NOT NULL,
    Audience_ID INTEGER NOT NULL,
    Channel_ID INTEGER NOT NULL,
    Date_Key INTEGER NOT NULL,
    Budget REAL NOT NULL,
    duration_days INTEGER NOT NULL,
    FOREIGN KEY (Type_ID) REFERENCES Dim_Campaign_Type(Type_ID),
    FOREIGN KEY (Audience_ID) REFERENCES Dim_Target_Audience(Audience_ID),
    FOREIGN KEY (Channel_ID) REFERENCES Dim_Channel(Channel_ID),
    FOREIGN KEY (Date_Key) REFERENCES Dim_Date(Date_Key)
);

-- 2. ส่วนดึงข้อมูลเตรียมเข้าโมเดล (Data Extraction & JOIN)
SELECT 
    f.Campaign_ID,
    ct.Campaign_Type,
    ta.Target_Audience,
    f.Budget,
    f.duration_days,
    ch.Channel_Name AS Channel
FROM Fact_Marketing_Campaign f
JOIN Dim_Campaign_Type ct    ON f.Type_ID = ct.Type_ID
JOIN Dim_Target_Audience ta  ON f.Audience_ID = ta.Audience_ID
JOIN Dim_Channel ch          ON f.Channel_ID = ch.Channel_ID
JOIN Dim_Date d              ON f.Date_Key = d.Date_Key;