/**
 * server.js
 * This is the main entry point for my server-side application.
 * It initializes an Express app, applies middleware (CORS, JSON body parsing),
 * and mounts the routes. Finally, it starts listening on a specified port.
 */

require('dotenv').config();  // Loads environment variables from .env
const express = require('express');
const passport = require('./config/passport'); // Import our Passport configuration
const cors = require('cors');
// Import routes
const authRoutes = require('./routes/authRoutes');
<<<<<<< Updated upstream
=======
const uploadRoutes = require('./routes/uploadRoutes');

>>>>>>> Stashed changes
const app = express();

// Initialize Passport (no sessions used)
app.use(passport.initialize());

// Middleware
app.use(cors());             // Enables Cross-Origin Resource Sharing
app.use(express.json());     // Parses incoming JSON requests

// attaches the routes from authRoutes.js to the path /api/auth.
app.use('/api/auth', authRoutes);

/**
 * Server Port Configuration
 * - Either use a PORT defined in .env or default to 5000
 */
const PORT = process.env.PORT || 5000;

/**
 * Start the Express server
 */
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
