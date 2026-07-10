# SafeGuide System Startup Commands

ssh jatayu@jatayu.local
passward ->jatayu@26
for disconectting cd then exit
Save this file for later! Whenever you restart your Raspberry Pi, you can use these commands to easily boot up both dashboards and deploy the Caretaker Portal to the internet.

### 1. Start Both Servers (Run on Raspberry Pi)

First, always make sure you are in the correct folder on the Pi:
```bash
cd /home/jatayu/freespace_navigation
```

**Command to start the AI Command Center (Port 5500):**
*This starts the camera feed, the object detection, the voice alerts, and the main dashboard. The `nohup` part means it will keep running in the background even if you close the terminal!*
```bash
nohup ./venv/bin/python app.py > app.log 2>&1 &
```

**Command to start the Caretaker Portal (Port 8000):**
*This serves the static HTML website for the Caretaker Portal.*
```bash
nohup python3 -m http.server 8000 -d caretaker-portal > portal.log 2>&1 &
```

*(Note: To kill them later if you need to, you can run `pkill -f app.py` and `pkill -f http.server`)*

---

### 2. Accessing them Locally (On your Wi-Fi)

- **AI Command Center (Main Dashboard):** Open your laptop browser and go to `http://10.134.165.237:5500`
- **Caretaker Portal:** Open your laptop browser and go to `http://10.134.165.237:8000`
**AI Command Center (Main Dashboard):** Open your laptop browser and go to http://10.159.27.237:5500
- **Caretaker Portal:** Open your laptop browser and go to http://10.159.27.237:8000

---

### 3. Deploying to the Internet (Run on Raspberry Pi)

To give someone anywhere in the world access to the Caretaker Portal, you need to tunnel Port 8000. 

First, ensure you have an SSH key generated on the Pi (you only have to do this once ever):
```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
```

Then, run this command to generate your live public link:
```bash
ssh -R 80:127.0.0.1:8000 localhost.run
```
*(Leave this terminal window open for as long as you want the website to be online!)*
