from datetime import datetime
from google.cloud import aiplatform


if __name__ == '__main__':
    TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

    job = aiplatform.PipelineJob(
        display_name="mlops-rf-diabetes",
        template_path="classification_pipeline.json",
        job_id="mlops-rf-diabetes-{0}".format(TIMESTAMP),
        location = "us-central1",
        enable_caching=False
    )

    job.submit()
    
    print('Pipeline successfully submitted')