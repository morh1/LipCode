// config/passport.js
const passport = require('passport');
const FacebookStrategy = require('passport-facebook').Strategy;
const GoogleStrategy = require('passport-google-oauth20').Strategy;
require('dotenv').config();
const { checkUserExists, createUser } = require('../services/authService');

// Configure Facebook Strategy
passport.use(
  new FacebookStrategy(
    {
      clientID: process.env.FACEBOOK_APP_ID,       // Your Facebook App ID
      clientSecret: process.env.FACEBOOK_APP_SECRET, // Your Facebook App Secret
      callbackURL: 'http://localhost:5000/api/auth/facebook/callback',
      profileFields: ['id', 'displayName', 'emails'], // Request the email field
    },
    async (accessToken, refreshToken, profile, done) => {
      try {
        // Extract the email from the Facebook profile
        const email = profile.emails && profile.emails[0].value;
        if (!email) {
          return done(new Error('No email provided by Facebook'));
        }

        // Check if the user exists in DynamoDB using your authService function
        const exists = await checkUserExists(email);
        if (!exists) {
          // If the user does not exist, create a new user
          await createUser(email);
        }

        // Pass the user information to the next middleware (here we only pass email and displayName)
        return done(null, { email, displayName: profile.displayName });
      } catch (error) {
        return done(error);
      }
    }
  )
);

// Configure Google Strategy
passport.use(
  new GoogleStrategy(
    {
      clientID: process.env.GOOGLE_CLIENT_ID,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      callbackURL: 'http://localhost:5000/api/auth/google/callback',
    },
    async (accessToken, refreshToken, profile, done) => {
      try {
        const email = profile.emails && profile.emails[0].value;
        if (!email) {
          return done(new Error('No email provided by Google'));
        }

        // Check if user exists in DynamoDB
        const exists = await checkUserExists(email);
        if (!exists) {
          await createUser(email);
        }

        return done(null, { email, displayName: profile.displayName });
      } catch (error) {
        return done(error);
      }
    }
  )
);

module.exports = passport;
