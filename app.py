
import gradio as gr
import pandas as pd
import joblib
import os

# Load the trained pipeline
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "customer_churn_model.joblib"
)
model = joblib.load(MODEL_PATH)

THRESHOLD = 0.35


def predict_churn(
    gender, senior, partner, dependents, tenure,
    phone, multiple_lines, internet, security, backup,
    protection, support, tv, movies, contract,
    paperless, payment, monthly, total
):
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": float(monthly),
        "TotalCharges": float(total)
    }])

    probability = float(model.predict_proba(customer)[0][1])

    if probability < THRESHOLD:
        risk = "Low Risk"
        color = "#15803d"
    elif probability < 0.60:
        risk = "Medium Risk"
        color = "#b45309"
    else:
        risk = "High Risk"
        color = "#b91c1c"

    result = f"""
    ### Prediction result

    **Estimated churn probability: {probability:.1%}**

    <div style="background:#e5e7eb;border-radius:10px;height:16px;">
      <div style="background:{color};width:{probability:.1%};
                  height:16px;border-radius:10px;"></div>
    </div>

    ### Risk category: {risk}

    *This is a model estimate, not a guarantee of customer behaviour.*
    """

    return result


service_options = ["Yes", "No", "No internet service"]

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Customer Churn Intelligence"
) as demo:

    gr.Markdown("""
    # Customer Churn Intelligence
    ### Predict customer churn risk using Machine Learning

    Enter a customer's service and billing details to estimate
    their probability of leaving the company.
    """)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Customer profile")

            gender = gr.Dropdown(
                ["Female", "Male"], value="Female", label="Gender"
            )
            senior = gr.Radio(
                [0, 1], value=0, label="Senior citizen (0 = No, 1 = Yes)"
            )
            partner = gr.Radio(
                ["Yes", "No"], value="No", label="Has a partner?"
            )
            dependents = gr.Radio(
                ["Yes", "No"], value="No", label="Has dependents?"
            )
            tenure = gr.Slider(
                0, 72, value=12, step=1, label="Tenure (months)"
            )

        with gr.Column():
            gr.Markdown("### Services")

            phone = gr.Radio(
                ["Yes", "No"], value="Yes", label="Phone service"
            )
            multiple_lines = gr.Dropdown(
                ["No", "Yes", "No phone service"],
                value="No", label="Multiple lines"
            )
            internet = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                value="Fiber optic", label="Internet service"
            )
            security = gr.Dropdown(
                service_options, value="No", label="Online security"
            )
            backup = gr.Dropdown(
                service_options, value="No", label="Online backup"
            )
            protection = gr.Dropdown(
                service_options, value="No", label="Device protection"
            )
            support = gr.Dropdown(
                service_options, value="No", label="Tech support"
            )
            tv = gr.Dropdown(
                service_options, value="No", label="Streaming TV"
            )
            movies = gr.Dropdown(
                service_options, value="No", label="Streaming movies"
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Contract and billing")

            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                value="Month-to-month", label="Contract"
            )
            paperless = gr.Radio(
                ["Yes", "No"], value="Yes", label="Paperless billing"
            )
            payment = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                value="Electronic check",
                label="Payment method"
            )
            monthly = gr.Number(
                value=70.0, minimum=0, label="Monthly charges"
            )
            total = gr.Number(
                value=840.0, minimum=0, label="Total charges"
            )

        with gr.Column():
            gr.Markdown("### Churn assessment")
            output = gr.Markdown(
                "Enter the details and click **Predict churn**."
            )

    predict_button = gr.Button(
        "Predict churn", variant="primary"
    )

    predict_button.click(
        fn=predict_churn,
        inputs=[
            gender, senior, partner, dependents, tenure,
            phone, multiple_lines, internet, security, backup,
            protection, support, tv, movies, contract,
            paperless, payment, monthly, total
        ],
        outputs=output
    )

    gr.Markdown("""
    ---
    **About:** Educational machine learning project using the
    Telco Customer Churn dataset and Logistic Regression.

    Risk categories use a preliminary 0.35 threshold.
    Predictions should not be treated as guaranteed outcomes.
    """)

if __name__ == "__main__":
    demo.launch()
