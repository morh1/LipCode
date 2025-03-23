/**
 * controllers/authController.js
 * Controllers handle the incoming HTTP requests and responses.
 * They parse input from req, call the service functions, and send out responses.
 * Call a service function for actual logic.
 */

const { checkUserExists, createUser } = require('../services/authService');
const nodemailer = require('nodemailer');


const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: process.env.MAIL_USER,      // e.g. "your@gmail.com"
    pass: process.env.MAIL_PASSWORD,  // e.g. "gmail_app_password"
  },
});

/**
 * 1) Check if user exists.
 * 2) If yes => respond { exists: true }.
 * 3) If no => send activation link => respond { exists: false }.
 */
exports.requestActivationLink = async (req, res) => {
  try {
    const { email } = req.body;
    const exists = await checkUserExists(email);

    if (exists) {
      // Already "activated"
      return res.json({ exists: true });
    }

    // Not in DB => send activation link
    const activationUrl = `http://localhost:5000/api/auth/activate?email=${encodeURIComponent(email)}`;

    await transporter.sendMail({
      from: process.env.MAIL_USER,
      to: email,
      subject: "Activate Your Account",
      text: `Hello! Please activate your account by clicking this link: ${activationUrl}`,
    });

    return res.json({
      exists: false,
      message: "Activation email sent!",
    });

  } catch (error) {
    console.error("Error requesting activation link:", error);
    return res.status(500).json({ error: "Server error" });
  }
};

/**
 * When the user clicks the activation link:
 *  - If not in DB => create them (activated by default).
 *  - If in DB => they've been activated already (or the email is taken).
 */
exports.activateUser = async (req, res) => {
  try {
    const { email } = req.query;

    if (!email) {
      return res.status(400).json({ error: "Email is required for activation" });
    }

    // Check if user is already in DB
    const exists = await checkUserExists(email);
    if (exists) {
      // Already "activated" need to take me to the upload page!!
      return res.redirect(`http://localhost:5173/dashboard?email=${encodeURIComponent(email)}`);
    }

    // Not in DB => create user => "activated"
    await createUser(email);

    res.send(`
      <html>
        <body style="text-align:center; margin-top:50px;">
          <h1>Account Activated!</h1>
          <p>Your account (${email}) was successfully created.</p>
          <a href="http://localhost:5173/dashboard?email=${encodeURIComponent(email)}">
            Go to our App
          </a>
        </body>
      </html>
    `);
  } catch (error) {
    console.error("Error activating user:", error);
    return res.status(500).json({ error: "Server error" });
  }
};

