/**
 * routes/authRoutes.js
 * Express Router that defines all authentication-related endpoints.
 * Each route calls a function from the 'authController'.
 */

const express = require('express');
const passport = require('passport');
const router = express.Router();
// Import controller functions
const authController = require('../controllers/authController');

/**
 * POST /api/auth/check-email
 *
 * - Calls requestActivationLink
 * - If user is in DB => { exists: true }
 * - If not => sends an email => { exists: false }
 */
router.post('/check-email', authController.requestActivationLink);

/**
 * GET /api/auth/activate?email=...
 * - If not in DB => create user => fully "activated"
 * - If in DB => do nothing or show "already activated"
 */
router.get('/activate', authController.activateUser);



/**
 * get /api/auth/facebook, get /api/auth/google
 *
 * - Instructs Passport to start the OAuth process by redirecting the user to Facebook’s of google's login.
 */
router.get('/facebook', passport.authenticate('facebook', { scope: ['email'] }));
router.get('/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

/**
 * get /api/auth/facebook/callback
 *
 * - Facebook redirects here after the user authorizes the app, and Passport handles the rest. You then decide what to do on success/failure.
 */
router.get(
  '/facebook/callback',
  passport.authenticate('facebook', { session: false, failureRedirect: 'http://localhost:5000/login' }),
  (req, res) => {
    // If authentication succeeded, req.user contains { email, displayName }
    // Redirect to the next page (dashboard) on the client
    // Here, we simply append the email as a query parameter; in a real app you might issue a token
    res.redirect(`http://localhost:5173/dashboard?email=${encodeURIComponent(req.user.email)}`);
  }
);
router.get('/google/callback',
  passport.authenticate('google', { session: false, failureRedirect: 'http://localhost:5000/login' }),
 (req, res) => {
    res.redirect(`http://localhost:5173/dashboard?email=${encodeURIComponent(req.user.email)}`);
  }
);
module.exports = router;
