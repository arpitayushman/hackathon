from dotenv import load_dotenv


load_dotenv()

from functools import wraps
from flask_cors import CORS
from flask import Flask, jsonify, Response, request, redirect, url_for
import flask
import os
from cache import MemoryCache

import vanna
from vanna.remote import VannaDefault

api_key = "afd2b0f7baa14ae3b506f6d74aa46960"
vanna_model_name = "rushalimodel123456"
vn = VannaDefault(model=vanna_model_name, api_key=api_key)

app = Flask(__name__, static_url_path="")

# SETUP
cache = MemoryCache()


vn.connect_to_postgres(
    host="localhost", dbname="test", user="postgres", password="root", port="5432"
)

df_information_schema = vn.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")

# This will break up the information schema into bite-sized chunks that can be referenced by the LLM
plan = vn.get_training_plan_generic(df_information_schema)
plan
vn.train(
    ddl="""
    CREATE TABLE IF NOT EXISTS public.bbps_fetch_txn_report(
    ref_id character varying(50) COLLATE pg_catalog."default" NOT NULL,
    txn_type character varying(20) COLLATE pg_catalog."default" NOT NULL,
    msg_id character varying(50) COLLATE pg_catalog."default" NOT NULL,
    mti character varying(30) COLLATE pg_catalog."default",
    blr_category character varying(100) COLLATE pg_catalog."default",
    response_code character varying(10) COLLATE pg_catalog."default",
    payment_channel character varying(30) COLLATE pg_catalog."default",
    cou_id character varying(20) COLLATE pg_catalog."default",
    bou_id character varying(20) COLLATE pg_catalog."default",
    bou_status character varying(20) COLLATE pg_catalog."default",
    cust_mobile_num character varying(20) COLLATE pg_catalog."default",
    tran_ref_id character varying(20) COLLATE pg_catalog."default",
    blr_id character varying(30) COLLATE pg_catalog."default",
    agent_id character varying(30) COLLATE pg_catalog."default",
    last_upd_host character varying(50) COLLATE pg_catalog."default",
    last_upd_port character varying(5) COLLATE pg_catalog."default",
    last_upd_site_cd character varying(5) COLLATE pg_catalog."default",
    crtn_ts timestamp without time zone,
    settlement_cycle_id character varying(30) COLLATE pg_catalog."default",
    complaince_cd character varying(20) COLLATE pg_catalog."default",
    complaince_reason character varying(100) COLLATE pg_catalog."default",
    mandatory_cust_params character varying(1000) COLLATE pg_catalog."default",
    initiating_ai character varying(10) COLLATE pg_catalog."default",
    txn_amount numeric,
    on_us character varying(1) COLLATE pg_catalog."default",
    payment_mode character varying(30) COLLATE pg_catalog."default",
    CONSTRAINT bbps_fetch_txn_report_pkey PRIMARY KEY (ref_id, txn_type)
    )
"""
)
vn.train(
    ddl="""
    CREATE TABLE public.bbps_ou_details ( ou_id varchar(4) NOT NULL, ou_type varchar(4) NULL, ou_name varchar(100) NULL, ou_effctv_from varchar(8) NULL, ou_effctv_to varchar(8) NULL, ou_endpoint_url varchar(100) NULL, ou_company_tan_no varchar(20) NULL, ou_uaadhar varchar(30) NULL, entity_status varchar(20) NOT NULL, crtn_ts timestamp NULL, crtn_user_id varchar(45) NULL, updt_ts timestamp NULL, updt_user_id varchar(45) NULL, gstin varchar(30) NULL, gst_state_code varchar(3) NULL, ou_group_name varchar(8) NULL, ndc_group varchar(4) NULL, is_parent_ou varchar(1) NULL, upms_request_response_format varchar(10) NULL, upms_end_point_url varchar(100) NULL, upms_supports_cou_reg_api bool NULL, upms_supports_blr_reg_api bool NULL, CONSTRAINT bbps_ou_details_pkey PRIMARY KEY (ou_id) );
"""
)
vn.train(
    ddl="""
    CREATE TABLE public.bbps_biller_details ( blr_id varchar(30) NOT NULL, blr_name varchar(100) NULL, blr_alias_name varchar(100) NULL, blr_category_name varchar(100) NULL, blr_linked_ou_default varchar(30) NULL, blr_linked_ou_backup_1 varchar(30) NULL, blr_linked_ou_backup_2 varchar(30) NULL, blr_linked_ou_backup_3 varchar(30) NULL, parent_blr_id varchar(30) NULL, is_parent_blr bpchar(1) NULL, blr_mode varchar(20) NULL, blr_accepts_adhoc varchar(1) NULL, blr_payment_modes json NULL, blr_payment_channels json NULL, blr_effctv_from varchar(8) NULL, blr_effctv_to varchar(8) NULL, blr_tan varchar(20) NULL, blr_uaadhaar varchar(30) NULL, blr_customer_params json NULL, entity_status varchar(20) NOT NULL, crtn_ts timestamp NULL, crtn_user_id varchar(45) NULL, updt_ts timestamp NULL, blr_pmt_amt_exactness varchar(10) NULL, fetch_requirement varchar(20) NULL, support_validation_api varchar(20) NULL, support_402_api_flag bool NULL, blr_timeout int2 NULL, blr_payment_modes_402 json NULL, plan_mdm_requirement varchar(20) NULL, upms_support_presentment_registration_flag bool NULL, CONSTRAINT bbps_biller_details_pkey PRIMARY KEY (blr_id) );

"""
)
vn.train(
    ddl="""
    CREATE TABLE public.bbps_certificates_details ( cert_id int8 NOT NULL, ou_id varchar(4) NULL, cert_type varchar(10) NULL, effctv_from varchar(8) NULL, effctv_to varchar(8) NULL, entity_status varchar(20) NOT NULL, crtn_ts timestamp NULL, crtn_user_id varchar(45) NULL, CONSTRAINT bbps_certificates_details_pkey PRIMARY KEY (cert_id, entity_status), CONSTRAINT fk_bbps_certificates_details_ou FOREIGN KEY (ou_id) REFERENCES public.bbps_ou_details (ou_id) ON DELETE SET NULL ON UPDATE CASCADE );
"""
)
training_data = vn.get_training_data()
training_data


#  YAHA tak
# NO NEED TO CHANGE ANYTHING BELOW THIS LINE
def requires_cache(fields):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            id = request.args.get("id")

            if id is None:
                return jsonify({"type": "error", "error": "No id provided"})

            for field in fields:
                if cache.get(id=id, field=field) is None:
                    return jsonify({"type": "error", "error": f"No {field} found"})

            field_values = {field: cache.get(id=id, field=field) for field in fields}

            # Add the id to the field_values
            field_values["id"] = id

            return f(*args, **field_values, **kwargs)

        return decorated

    return decorator


@app.route("/api/v0/generate_questions", methods=["GET"])
def generate_questions():
    return jsonify(
        {
            "type": "question_list",
            "questions": vn.generate_questions(),
            "header": "Here are some questions you can ask:",
        }
    )

app = Flask(__name__, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}},supports_credentials=True) 
@app.route("/api/v0/generate_sql", methods=["GET"])
def generate_sql():
    question = flask.request.args.get("question")

    if question is None:
        return jsonify({"type": "error", "error": "No question provided"})

    id = cache.generate_id(question=question)
    sql = vn.generate_sql(question=question)

    cache.set(id=id, field="question", value=question)
    cache.set(id=id, field="sql", value=sql)

    return jsonify(
        {
            "type": "sql",
            "id": id,
            "text": sql,
        }
    )


@app.route("/api/v0/run_sql", methods=["GET"])
@requires_cache(["sql"])
def run_sql(id: str, sql: str):
    try:
        df = vn.run_sql(sql=sql)

        cache.set(id=id, field="df", value=df)

        return jsonify(
            {
                "type": "df",
                "id": id,
                "df": df.head(10).to_json(orient="records"),
            }
        )

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})


@app.route("/api/v0/download_csv", methods=["GET"])
@requires_cache(["df"])
def download_csv(id: str, df):
    csv = df.to_csv()

    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={id}.csv"},
    )


@app.route("/api/v0/generate_plotly_figure", methods=["GET"])
@requires_cache(["df", "question", "sql"])
def generate_plotly_figure(id: str, df, question, sql):
    print(df)
    try:
        code = vn.generate_plotly_code(
            question=question,
            sql=sql,
            df_metadata=f"Running df.dtypes gives:\n {df.dtypes}",
        )
        fig = vn.get_plotly_figure(plotly_code=code, df=df, dark_mode=False)
        fig_json = fig.to_json()

        cache.set(id=id, field="fig_json", value=fig_json)

        return jsonify(
            {
                "type": "plotly_figure",
                "id": id,
                "fig": fig_json,
            }
        )
    except Exception as e:
        # Print the stack trace
        import traceback

        traceback.print_exc()

        return jsonify({"type": "error", "error": str(e)})


@app.route("/api/v0/get_training_data", methods=["GET"])
def get_training_data():
    df = vn.get_training_data()

    return jsonify(
        {
            "type": "df",
            "id": "training_data",
            "df": df.head(25).to_json(orient="records"),
        }
    )


@app.route("/api/v0/remove_training_data", methods=["POST"])
def remove_training_data():
    # Get id from the JSON body
    id = flask.request.json.get("id")

    if id is None:
        return jsonify({"type": "error", "error": "No id provided"})

    if vn.remove_training_data(id=id):
        return jsonify({"success": True})
    else:
        return jsonify({"type": "error", "error": "Couldn't remove training data"})


@app.route("/api/v0/train", methods=["POST"])
def add_training_data():
    question = flask.request.json.get("question")
    sql = flask.request.json.get("sql")
    ddl = flask.request.json.get("ddl")
    documentation = flask.request.json.get("documentation")

    try:
        id = vn.train(question=question, sql=sql, ddl=ddl, documentation=documentation)

        return jsonify({"id": id})
    except Exception as e:
        print("TRAINING ERROR", e)
        return jsonify({"type": "error", "error": str(e)})


@app.route("/api/v0/generate_followup_questions", methods=["GET"])
@requires_cache(["df", "question", "sql"])
def generate_followup_questions(id: str, df, question, sql):
    followup_questions = vn.generate_followup_questions(
        question=question, sql=sql, df=df
    )

    cache.set(id=id, field="followup_questions", value=followup_questions)

    return jsonify(
        {
            "type": "question_list",
            "id": id,
            "questions": followup_questions,
            "header": "Here are some followup questions you can ask:",
        }
    )


@app.route("/api/v0/load_question", methods=["GET"])
@requires_cache(["question", "sql", "df", "fig_json", "followup_questions"])
def load_question(id: str, question, sql, df, fig_json, followup_questions):
    try:
        return jsonify(
            {
                "type": "question_cache",
                "id": id,
                "question": question,
                "sql": sql,
                "df": df.head(10).to_json(orient="records"),
                "fig": fig_json,
                "followup_questions": followup_questions,
            }
        )

    except Exception as e:
        return jsonify({"type": "error", "error": str(e)})


@app.route("/api/v0/get_question_history", methods=["GET"])
def get_question_history():
    return jsonify(
        {
            "type": "question_history",
            "questions": cache.get_all(field_list=["question"]),
        }
    )


@app.route("/")
def root():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(
        debug=True,
    )
