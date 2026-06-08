// if (!sessionStorage.getItem('nav_auth')) {
//     window.location.href = 'login.html';
//   }

// //─── FLASK INTEGRATION (RECOMMENDED FOR PRODUCTION) ─────────────────────────
// For stronger auth, add a Flask login route instead of client-side checks:

//   from flask import session, redirect, url_for, request
//   app.secret_key = 'your-secret-key-here'

//   CREDENTIALS = {"admin": "blind2024"}   # move to env var or hashed store

//   @app.route('/login', methods=['GET','POST'])
//   def login():
//       if request.method == 'POST':
//           u = request.form.get('username')
//           p = request.form.get('password')
//           if CREDENTIALS.get(u) == p:
//               session['user'] = u
//               return redirect(url_for('index'))
//           return render_template('login.html', error=True)
//       return render_template('login.html')

//   @app.route('/logout')
//   def logout():
//       session.clear()
//       return redirect(url_for('login'))

//   # Protect your index route:
//   @app.route('/')
//   def index():
//       if 'user' not in session:
//           return redirect(url_for('login'))
//       return send_from_directory(UI_FOLDER, 'index.html')
