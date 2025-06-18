# AWS Deployment Guide

This guide explains how to deploy the Flask Trading Algorithm to AWS.

## Prerequisites

- AWS Account with appropriate permissions
- AWS CLI installed and configured
- Docker (for containerized deployment)
- Git repository with your application code

## Deployment Options

### 1. AWS Elastic Beanstalk (Recommended for beginners)

Elastic Beanstalk is the easiest way to deploy Flask applications on AWS.

#### Windows Deployment (Recommended)

Due to EB CLI compatibility issues on Windows, use the manual deployment method:

1. **Run the deployment preparation script:**
   ```cmd
   cd 0_Launchers/flask_web
   python create_deployment_zip.py
   ```

2. **Follow the manual deployment steps:**
   - Go to [AWS Elastic Beanstalk Console](https://console.aws.amazon.com/elasticbeanstalk/)
   - Click "Create Application"
   - Application name: `trading-algorithm-web`
   - Platform: Python
   - Platform branch: Python 3.9
   - Platform version: 3.9.16 (latest)
   - Click "Configure more options"
   - Environment type: Single instance (free tier)
   - Instance type: t3.micro (free tier) or t3.medium
   - Click "Create environment"
   - Once created, go to "Upload and deploy"
   - Upload the `deployment.zip` file created by the script
   - Click "Deploy"
   - Wait 5-10 minutes for deployment to complete

#### Alternative: AWS CLI Deployment

If you have AWS CLI configured:

1. **Run the AWS CLI deployment script:**
   ```cmd
   cd 0_Launchers/flask_web
   deploy_with_aws_cli.bat
   ```

2. **Use the S3 URL provided by the script in Elastic Beanstalk**

#### Linux/Mac Deployment (EB CLI)

If you're on Linux or Mac, you can use the EB CLI:

1. **Prepare your application:**
   ```bash
   cd 0_Launchers/flask_web
   # Ensure all files are committed to git
   git add .
   git commit -m "Prepare for EB deployment"
   ```

2. **Install EB CLI:**
   ```bash
   pip install awsebcli
   ```

3. **Initialize EB application:**
   ```bash
   eb init trading-algorithm-web
   # Follow the prompts to select region, platform, etc.
   ```

4. **Create environment:**
   ```bash
   eb create trading-algorithm-prod
   ```

5. **Deploy:**
   ```bash
   eb deploy
   ```

6. **Open the application:**
   ```bash
   eb open
   ```

#### Configuration:
- The `.ebextensions/01_flask.config` file configures the environment
- The `Procfile` tells EB how to run the application
- Environment variables are set in the EB configuration

### 2. AWS EC2 (Manual deployment)

For more control over the deployment process.

#### Steps:

1. **Launch an EC2 instance:**
   - Use Ubuntu 20.04 LTS
   - t3.medium or larger recommended
   - Configure security groups to allow HTTP (80) and HTTPS (443)

2. **Connect to your instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. **Deploy the application:**
   ```bash
   # Option 1: Use Docker (recommended)
   docker build -t trading-algorithm-web .
   docker run -d -p 80:5000 trading-algorithm-web
   
   # Option 2: Manual deployment
   # Copy application files to /opt/trading-algorithm-web
   # Set up virtual environment and install dependencies
   # Configure Nginx and Supervisor manually
   ```

4. **Access your application:**
   - Open `http://your-instance-ip` in your browser

### 3. AWS ECS (Containerized deployment)

For containerized deployment using Docker.

#### Steps:

1. **Build and push Docker image:**
   ```bash
   # Build the image
   docker build -t trading-algorithm-web .
   
   # Tag for ECR
   docker tag trading-algorithm-web:latest your-account.dkr.ecr.region.amazonaws.com/trading-algorithm-web:latest
   
   # Push to ECR
   aws ecr get-login-password --region region | docker login --username AWS --password-stdin your-account.dkr.ecr.region.amazonaws.com
   docker push your-account.dkr.ecr.region.amazonaws.com/trading-algorithm-web:latest
   ```

2. **Create ECS cluster and service:**
   - Use the AWS Console or AWS CLI
   - Configure task definition with the ECR image
   - Set up Application Load Balancer

3. **Deploy:**
   - Update the service with the new image
   - ECS will handle the rolling deployment

### 4. AWS Lambda + API Gateway (Serverless)

For serverless deployment (limited WebSocket support).

#### Steps:

1. **Install Zappa:**
   ```bash
   pip install zappa
   ```

2. **Initialize Zappa:**
   ```bash
   zappa init
   ```

3. **Deploy:**
   ```bash
   zappa deploy production
   ```

## Environment Variables

Set these environment variables in your deployment:

```bash
FLASK_ENV=production
SECRET_KEY=your-secure-secret-key
LOG_LEVEL=INFO
CORS_ORIGINS=your-domain.com
MODEL_SAVE_PATH=/tmp/trading_model
DATA_CACHE_DIR=/tmp/stock_data
```

## Security Considerations

1. **HTTPS/SSL:**
   - Use AWS Certificate Manager for SSL certificates
   - Configure HTTPS redirects

2. **Secrets Management:**
   - Use AWS Secrets Manager for sensitive data
   - Never commit secrets to version control

3. **Network Security:**
   - Configure security groups properly
   - Use VPC for network isolation

4. **Application Security:**
   - Keep dependencies updated
   - Use strong secret keys
   - Enable CORS properly

## Monitoring and Logging

1. **CloudWatch Logs:**
   - Configure log groups for application logs
   - Set up log retention policies

2. **CloudWatch Metrics:**
   - Monitor CPU, memory, and network usage
   - Set up alarms for critical metrics

3. **Application Monitoring:**
   - Use AWS X-Ray for tracing
   - Monitor response times and errors

## Scaling

1. **Auto Scaling:**
   - Configure auto scaling groups
   - Set up scaling policies based on metrics

2. **Load Balancing:**
   - Use Application Load Balancer
   - Configure health checks

3. **Database Scaling:**
   - Use RDS for persistent data
   - Configure read replicas if needed

## Cost Optimization

1. **Instance Types:**
   - Use appropriate instance types
   - Consider spot instances for non-critical workloads

2. **Reserved Instances:**
   - Purchase reserved instances for predictable workloads
   - Use savings plans for cost optimization

3. **Monitoring:**
   - Monitor costs using AWS Cost Explorer
   - Set up billing alerts

## Troubleshooting

### Common Issues:

1. **Application not starting:**
   - Check logs: `eb logs` or CloudWatch logs
   - Verify environment variables
   - Check port configuration

2. **WebSocket issues:**
   - Ensure proxy configuration supports WebSockets
   - Check CORS settings
   - Verify SocketIO configuration

3. **Performance issues:**
   - Monitor resource usage
   - Check database connections
   - Optimize application code

### Useful Commands:

```bash
# Elastic Beanstalk
eb logs
eb status
eb health

# EC2
sudo supervisorctl status
sudo tail -f /var/log/nginx/error.log
sudo systemctl status nginx

# Docker
docker logs container-name
docker exec -it container-name bash
```

## Maintenance

1. **Regular Updates:**
   - Keep dependencies updated
   - Apply security patches
   - Monitor for vulnerabilities

2. **Backup Strategy:**
   - Backup application data
   - Backup configuration files
   - Test restore procedures

3. **Monitoring:**
   - Set up health checks
   - Monitor application performance
   - Track user metrics

## Support

For issues specific to this application:
- Check the application logs
- Review the configuration files
- Test locally before deploying

For AWS-specific issues:
- Check AWS documentation
- Use AWS Support if needed
- Monitor AWS Service Health Dashboard 