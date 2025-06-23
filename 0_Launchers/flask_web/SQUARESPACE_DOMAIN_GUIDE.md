# Squarespace Domain Setup Guide for Trading Algorithm

## Overview
This guide will help you set up a subdomain on your Squarespace-managed domain `mylensandi.com` to point to your EC2 instance at `35.89.15.141`.

## Current Setup
- **Domain:** `mylensandi.com` (managed by Squarespace)
- **EC2 Instance:** `35.89.15.141`
- **Target Subdomain:** `trading.mylensandi.com` (or your choice)

## Important: Squarespace Limitations

Squarespace has different DNS management capabilities depending on your plan:

### Available Plans and DNS Access:
- **Personal Plan:** Limited DNS management
- **Business Plan:** Basic DNS management
- **Commerce Plans:** Full DNS management
- **Enterprise Plan:** Full DNS management

## Method 1: DNS Records (Recommended - If Available)

### Step 1: Check Your Squarespace Plan
1. Log into your Squarespace account
2. Go to **Settings** → **Billing & Account**
3. Check your current plan level

### Step 2: Access DNS Settings
1. Go to **Settings** → **Domains**
2. Click on `mylensandi.com`
3. Look for **"DNS Settings"** or **"Advanced DNS"**

### Step 3: Add DNS Record
If DNS Records are available:
1. Click **"Add Record"**
2. **Record Type:** A
3. **Name:** `trading` (or your desired subdomain)
4. **Points to:** `35.89.15.141`
5. **TTL:** `300` (or leave default)
6. Click **"Save"**

## Method 2: External Redirect (If DNS Records Not Available)

### Step 1: Create a Hidden Page
1. Go to **Pages** in your Squarespace site
2. Click **"+"** to add a new page
3. Name it `trading` (or your desired subdomain)
4. Set it as **"Not Linked"** (hidden from navigation)

### Step 2: Add External Link
1. Click on the new page
2. Add a **"Link"** block
3. Set the link to: `http://35.89.15.141`
4. Save the page

### Step 3: Set Up URL Redirect
1. Go to **Settings** → **Domains**
2. Click on `mylensandi.com`
3. Look for **"URL Mappings"** or **"Redirects"**
4. Add a redirect:
   - **From:** `/trading`
   - **To:** `http://35.89.15.141`
   - **Type:** 301 (permanent)

## Method 3: Contact Squarespace Support

If neither method works, contact Squarespace support:

1. Go to **Help Center**
2. Search for "DNS management" or "subdomain setup"
3. Contact support and request:
   - DNS management access for your plan
   - Help setting up a subdomain pointing to external IP
   - Alternative solutions for your use case

## Method 4: Domain Transfer (Last Resort)

If Squarespace limitations prevent your needs:

### Transfer to Another Registrar:
1. **GoDaddy:** $12-15/year, full DNS control
2. **Namecheap:** $10-12/year, full DNS control
3. **Google Domains:** $12/year, full DNS control
4. **AWS Route 53:** $12/year + $0.50/month hosted zone

### Transfer Process:
1. Unlock domain in Squarespace
2. Get transfer authorization code
3. Initiate transfer with new registrar
4. Approve transfer in Squarespace
5. Set up DNS records with new registrar

## Testing Your Setup

### Step 1: DNS Propagation Check
```bash
nslookup trading.mylensandi.com
```
Expected result: Should resolve to `35.89.15.141`

### Step 2: HTTP Connectivity Test
```bash
curl -I http://trading.mylensandi.com
```
Expected result: Should return HTTP 200 OK

### Step 3: Application Test
1. Open browser: `http://trading.mylensandi.com`
2. Verify your trading algorithm interface loads
3. Test all functionality

## Troubleshooting

### DNS Not Working?
1. **Check Squarespace plan:** Ensure you have DNS management access
2. **Verify DNS record:** Ensure A record points to `35.89.15.141`
3. **Wait for propagation:** DNS can take 5-60 minutes
4. **Clear browser cache:** Try incognito/private browsing

### Application Not Loading?
1. **Check EC2 security group:** Ensure port 80 is open
2. **Verify nginx is running:** SSH into EC2 and check services
3. **Check application logs:** Look for errors in Flask/nginx logs

### Squarespace-Specific Issues?
1. **Plan limitations:** Upgrade to Business or Commerce plan
2. **DNS not available:** Use redirect method or contact support
3. **Subdomain conflicts:** Ensure subdomain name is unique

## Alternative Solutions

### Option 1: Use a Different Subdomain
- `algo.mylensandi.com`
- `neural.mylensandi.com`
- `ai.mylensandi.com`
- `stock.mylensandi.com`

### Option 2: Use a Subdirectory
- `mylensandi.com/trading`
- `mylensandi.com/algo`
- `mylensandi.com/neural`

### Option 3: Free Subdomain Service
- Use a free service like Freenom (.tk domain)
- Point it to your EC2 instance
- Link from your main site

## Cost Considerations

### Current Setup:
- **Domain:** Already owned (mylensandi.com)
- **EC2 Instance:** Your existing costs
- **Squarespace Plan:** Your existing costs

### Potential Additional Costs:
- **Squarespace Plan Upgrade:** $18-40/month (if needed for DNS)
- **Domain Transfer:** $10-15 (one-time, if transferring)
- **SSL Certificate:** Free with Let's Encrypt

## Security Considerations

### HTTPS Setup (Recommended):
1. SSH into your EC2 instance
2. Install Certbot: `sudo yum install -y certbot python3-certbot-nginx`
3. Get SSL certificate: `sudo certbot --nginx -d trading.mylensandi.com`
4. Set up auto-renewal: `sudo certbot renew --dry-run`

### Security Best Practices:
1. Use HTTPS instead of HTTP
2. Keep EC2 security groups minimal
3. Regular security updates
4. Monitor access logs

## Next Steps After Setup

1. **Test thoroughly:** Ensure all functionality works
2. **Set up monitoring:** Monitor domain and application health
3. **Create documentation:** Document the setup for future reference
4. **Consider HTTPS:** Set up SSL certificate for security
5. **Backup strategy:** Regular backups of your application

## Support Resources

- **Squarespace Help Center:** https://support.squarespace.com
- **Squarespace DNS Documentation:** Search "DNS management" in help center
- **AWS EC2 Documentation:** https://docs.aws.amazon.com/ec2/
- **Nginx Documentation:** https://nginx.org/en/docs/

## Quick Commands

### Check DNS Propagation:
```bash
nslookup trading.mylensandi.com
dig trading.mylensandi.com
```

### Test HTTP Connectivity:
```bash
curl -I http://trading.mylensandi.com
wget --spider http://trading.mylensandi.com
```

### Check EC2 Instance:
```bash
ssh -i "your-key.pem" ec2-user@35.89.15.141
sudo systemctl status nginx
sudo systemctl status trading-algorithm
```

Remember: The key is to work within Squarespace's limitations while achieving your goal of having a professional subdomain for your trading algorithm. 