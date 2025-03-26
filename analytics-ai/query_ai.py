from dotenv import load_dotenv

from vanna.flask import VannaFlaskApp

# Frontend path is -> # path for frontend -> C:\Users\localadmin\anaconda3\Lib\site-packages\vanna\flask\auth.py
load_dotenv()

from functools import wraps
from flask_cors import CORS
from flask import Flask, jsonify, Response, request, redirect, url_for
import flask
import os
from cache import MemoryCache

import vanna
from vanna.remote import VannaDefault

api_key = "51eed2013eaa421580f1f6ffa40f5d9c"
vanna_model_name = "akashg"
vn = VannaDefault(model=vanna_model_name, api_key=api_key)

# app = Flask(__name__, static_url_path="")


# SETUP
cache = MemoryCache()


vn.connect_to_postgres(
    host="localhost", dbname="test", user="postgres", password="root", port="5432"
)

df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
plan
# vn.train(
#     ddl="""
#     CREATE TABLE IF NOT EXISTS public.bbps_txn_report
# (
#     ref_id character varying(50) COLLATE pg_catalog."default" NOT NULL,
#     txn_type character varying(20) COLLATE pg_catalog."default" NOT NULL,
#     msg_id character varying(50) COLLATE pg_catalog."default" NOT NULL,
#     mti character varying(30) COLLATE pg_catalog."default",
#     blr_category character varying(100) COLLATE pg_catalog."default",
#     response_code character varying(10) COLLATE pg_catalog."default",
#     payment_channel character varying(30) COLLATE pg_catalog."default",
#     cou_id character varying(20) COLLATE pg_catalog."default",
#     bou_id character varying(20) COLLATE pg_catalog."default",
#     bou_status character varying(20) COLLATE pg_catalog."default",
#     cust_mobile_num character varying(20) COLLATE pg_catalog."default",
#     tran_ref_id character varying(20) COLLATE pg_catalog."default",
#     blr_id character varying(30) COLLATE pg_catalog."default",
#     agent_id character varying(30) COLLATE pg_catalog."default",
#     last_upd_host character varying(50) COLLATE pg_catalog."default",
#     last_upd_port character varying(5) COLLATE pg_catalog."default",
#     last_upd_site_cd character varying(5) COLLATE pg_catalog."default",
#     crtn_ts timestamp without time zone,
#     settlement_cycle_id character varying(30) COLLATE pg_catalog."default",
#     complaince_cd character varying(20) COLLATE pg_catalog."default",
#     complaince_reason character varying(100) COLLATE pg_catalog."default",
#     mandatory_cust_params character varying(1000) COLLATE pg_catalog."default",
#     initiating_ai character varying(10) COLLATE pg_catalog."default",
#     txn_amount numeric,
#     on_us character varying(1) COLLATE pg_catalog."default",
#     payment_mode character varying(30) COLLATE pg_catalog."default",
#     status character varying(20) COLLATE pg_catalog."default",
#     CONSTRAINT bbps_txn_report_pkey PRIMARY KEY (ref_id, txn_type)
# )
# """
# )
training_data = vn.get_training_data()
training_data


app = VannaFlaskApp(vn, allow_llm_to_see_data=True)
app.run()