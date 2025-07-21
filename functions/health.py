# Lambda handler for health check endpoint
# This will be implemented in TASK-032

def handler(event, context):
    """
    Health check endpoint handler
    """
    # TODO: Implement in TASK-032
    return {
        'statusCode': 200,
        'body': '{"status": "healthy", "message": "Service is running"}'
    }