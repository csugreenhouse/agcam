import cv2
import numpy as np
import sys 

sys.path.append('/mnt/db/agcam')

from datetime import datetime
import psycopg2
import os
import json

def open_connection_to_database():
    with open("/mnt/db/agcam/utils/secrets_util.json") as f:
        secrets = json.load(f)["database"]
    try:
        conn = psycopg2.connect(
            host=secrets["PG_HOST"],
            dbname=secrets["PG_DATABASE"],
            user=secrets["PG_USER"],
            password=secrets["PG_PASSWORD"]
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None
        conn.close() 
    
def close_connection_to_database(conn):
    if conn:
        conn.close() 

def append_photo_to_imgIndex(conn, cam_id, file_path):
    query = (
        "INSERT INTO imgIndex (cam_id, file_path) "
        "VALUES (%s, %s);"
    )
    try:
        execute_query(conn, query, (cam_id, file_path))
    except Exception as e:
        print(f"Error appending photo to imgIndex: {e}")

def print_imgIndex_in_terminal(conn):   
    query = (
        "SELECT * FROM imgIndex;"
    )
    results = execute_query(conn, query)
    for row in results:
        print(f"cam_id: {row[0]}, file_path: {row[1]}")
    
def query_to_CSV_file(results, CSVfile_path):
    import csv
    with open(CSVfile_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(results[0].keys())
        for entry in results:
            writer.writerow(entry.values())
        print(CSVfile_path)

def execute_query(conn,query,params=None):
    try:
        cur = conn.cursor()
        cur.execute(query,params)
        if cur.description is not None:
            results = cur.fetchall()
        else:
            results = []
        conn.commit()
        cur.close()
        return results
    except Exception as e:
        conn.rollback()
        raise LookupError(f"Database error: {e}")
    finally:
        pass