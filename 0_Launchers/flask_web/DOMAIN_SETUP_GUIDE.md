# Domain Setup Guide for Trading Algorithm

## Overview
This guide will help you set up a custom domain name (like `trading-algorithm.com`) for your EC2-hosted trading algorithm instead of using the IP address.

## Current Setup
- **EC2 Instance:** `i-0d713dc81235dd231`
- **Current IP:** `35.89.15.141`
- **Region:** `us-west-2`

## Step 1: Choose and Register a Domain

### Recommended Domain Registrars:
1. **AWS Route 53** (Best integration with EC2)
2. **Namecheap** (Good prices, easy to use)
3. **GoDaddy** (Popular, good support)
4. **Google Domains** (Clean interface)

### Suggested Domain Names:
- `trading-algorithm.com`
- `kitkumar-trading.com`
- `neural-trading.com`
- `ai-trading-system.com`
- `trading-ai.com`

## Step 2: Configure DNS Records

### Option A: AWS Route 53 (Recommended)

1. **Register domain through Route 53** (if not already done)
2. **Create Hosted Zone:**
   - Go to Route 53 → Hosted Zones
   - Click "Create Hosted Zone"
   - Enter your domain name
   - Click "Create"

3. **Create A Record:**
   - Click on your hosted zone
   - Click "Create Record"
   - **Record Type:** A
   - **Record Name:** Leave blank (for root domain)
   - **Value:** `35.89.15.141`
   - **TTL:** `300`
   - Click "Create Records"

### Option B: Other Domain Registrars

1. **Access DNS Settings:**
   - Log into your domain registrar
   - Find DNS management or DNS settings
   - Look for "A Records" or "DNS Records"

2. **Create A Record:**
   - **Host/Name:** `@` (or leave blank for root domain)
   - **Points to/Value:** `35.89.15.141`
   - **TTL:** `300` (5 minutes)

## Step 3: Update Configuration

After setting up DNS, update your configuration:

1. **Edit `ec2_config.txt`:**
   ```
   # Change this line:
   PUBLIC_IP=35.89.15.141
   
   # To your domain:
   PUBLIC_IP=your-domain.com
   ```

2. **Update any scripts or documentation** that reference the IP address

## Step 4: Test Your Domain

1. **Wait for DNS Propagation** (5-60 minutes)
2. **Test the domain:** `http://your-domain.com`
3. **Verify all functionality works**

## Step 5: Optional - Set Up HTTPS

For security and professionalism, consider setting up HTTPS:

### Using Let's Encrypt (Free):
```bash
# SSH into your EC2 instance
ssh -i "your-key.pem" ec2-user@35.89.15.141

# Install Certbot
sudo yum install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Test auto-renewal
sudo certbot renew --dry-run
```

## Step 6: Update Application Configuration

If your Flask app needs to know about the domain:

1. **Update `config.py`:**
   ```python
   # Add domain configuration
   DOMAIN_NAME = 'your-domain.com'
   BASE_URL = f'https://{DOMAIN_NAME}'  # or http:// for non-HTTPS
   ```

2. **Update any hardcoded URLs** in your application

## Troubleshooting

### DNS Not Working?
1. **Check DNS propagation:** Use `nslookup your-domain.com`
2. **Verify A record:** Ensure it points to `35.89.15.141`
3. **Wait longer:** DNS can take up to 48 hours (usually 5-60 minutes)

### Application Not Loading?
1. **Check EC2 security group:** Ensure port 80 (and 443 for HTTPS) are open
2. **Verify nginx configuration:** Check `/etc/nginx/conf.d/trading-algorithm.conf`
3. **Check application logs:** `sudo journalctl -u trading-algorithm -f`

### HTTPS Issues?
1. **Check firewall:** Ensure port 443 is open
2. **Verify certificate:** `sudo certbot certificates`
3. **Check nginx config:** `sudo nginx -t`

## Cost Considerations

- **Domain Registration:** $10-15/year
- **Route 53 Hosted Zone:** $0.50/month (if using Route 53)
- **SSL Certificate:** Free with Let's Encrypt
- **EC2 Instance:** Your existing costs

## Benefits of Custom Domain

1. **Professional appearance:** `trading-algorithm.com` vs `35.89.15.141`
2. **Easy to remember:** Users can bookmark and share easily
3. **Branding:** Establishes your trading algorithm as a product
4. **Flexibility:** Can move to different servers without changing URLs
5. **SEO:** Better for search engine optimization

## Next Steps After Domain Setup

1. **Set up monitoring:** Monitor domain and application health
2. **Create subdomains:** Consider `api.trading-algorithm.com` for API endpoints
3. **Set up email:** Professional email addresses like `admin@trading-algorithm.com`
4. **Backup strategy:** Regular backups of your application and data
5. **Documentation:** Update all documentation with new domain

## Support

If you encounter issues:
1. Check AWS Route 53 documentation
2. Review nginx and Flask logs
3. Test connectivity step by step
4. Consider using AWS support if needed 