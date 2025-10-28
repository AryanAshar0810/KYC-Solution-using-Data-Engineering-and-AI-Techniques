<h1>Project Overview</h1>
This project implements a simplified yet production-ready Know Your Client (KYC) solution in Azure, which ingests, processes, analyses, and serves client and transaction data. The system combines business rule-based risk scoring, machine learning anomaly detection, and multi-agent consensus mechanisms to provide comprehensive compliance and risk assessment.

<h1>Execution of the project</h1>
<ul>
  <li>Install all the libraries listed in “requirements.txt” </li>
  <li> Delete all the output folders if you have installed them from GitHub</li>
  <ul>
    <li>data</li>
    <li>final_outout_dataset</li>
    <li>model</li>
  </ul>
<li>Running the Python script titled “main_pipeline.py”</li>
</ul>

<b> THE DETAILED DESCRIPTION OF THE DATASET AS WELL AS THE ANSWERS TO ALL THE TASKS CAN BE FOUND IN THE USER GUIDE. FOR MORE INFORMATION AND ANSWERS OF EACH INDIVIDUAL TASK, REFER TO THE USER GUIDE TITLED "User Guide - KYC Solution in Azure and Databricks.pdf". </b> <br> <br>
<b>PLEASE NOTE THAT ALL THE TASKS HAVE BEEN ANSWERED, INCLUDING TASKS FROM 2.1 TO 2.9 (BOTH OPTIONAL BONUS QUESTIONS HAVE BEEN ANSWERED) </b>

<h1>Design Decisions</h1>
<ul>
<li>Two-Tier Risk Scoring (Business Rules + ML): ML only approach is less suitable and interpretable for regulatory audits. Business rules provide transparency and explainability, and ML models detect novel patterns. This combination leverages the strengths of both approaches.</li>
<li>Isolation Forest for Anomaly Detection: I used isolation forest instead of Logistic Regression and Random Forest Classifiers since it is an unsupervised learning method, so no labelled training data is required and making it efficient for high-dimensional data and robust to multicollinearity. </li>
<li> Contamination Rate of 0.2 for Clients, 0.3 for Transactions: Transactions are more frequent and naturally noisier than client profiles. Higher contamination on transactions reflects realistic fraud distribution (~30% suspicious)and lower contamination on clients reflects KYC strictness (~20% anomalous). These are assumptions; real systems would calibrate based on historical data.</li>
</ul>

<h1>Trade-offs</h1> 
<ul>
<li>Higher contamination rates (0.2-0.3) flag more items as anomalous, increasing false positives. In KYC/AML systems, false negatives (missing actual risk) are worse than false positives (over-flagging). With the multi-agent code, I can also use confidence scores and multi-agent consensus to reduce false positives.</li>
<li>I have one Isolation Forest per entity type, not an ensemble of multiple models. This leads to faster deployment and inference and makes outputs easier to explain; however, it can have lower accuracy than an ensemble. In future, I could ensemble Isolation Forest + Local Outlier Factor + One-Class SVM for higher accuracy.</li>
<li>In risk scoring, I have additive risk scoring rather than multiplicative. This means business rule scores are additive (risks compound) rather than multiplicative. This is because an additive can lead to becoming more conservative (multiple flags = higher risk), whereas a multiplicative would amplify scores too aggressively.
</li>
</ul>

<h1>Future Improvements</h1>
<ul>
<li>Behavioral Profiling & Anomaly Detection: An enhancement can be to learn baseline behaviour for each client and flag deviations. Metrics which can be used are typical transaction amounts, frequencies, destinations and seasonal spending patterns. It can be implemented with historical transactions of the clients and behaviour changes can suggest something not going well. </li>
<li>External Data Integration: Connection with third-party APIs can help for data enrichment like UN database for sanctions, credit bureaus for credit history and fraud reports etc. It can be implemented by Azure Logic Apps and rate limiting to respect API quotas. </li>
<li>Explainable AI Integration: I can add SHAP or LIME explanations for individual ML predictions. SHAP value will show which features most contributed to high risk score.</li> 
<li>Learn transaction patterns using deep learning models like graph neural networks
<li>Adversarial KYC Game-Playing Detection: We can detect clients deliberately gaming the system (structuring, smurfing, round-tripping). Some unique patterns are multiple deposits just below threshold, coordinated small transactions from multiple related clients converging to a single account, money sent out then returned within days (circular flow) and rapid pattern changes when the client senses scrutiny. It can be implemented by correlation analysis, identifying temporal anomalies and statistical burst detection.

Please refer to the user guide for more information on datasets and a detailed answer to all the tasks in the technical assessment. 
