require('dotenv').config();
const express = require('express');
const passport = require('./config/passport');
const cors = require('cors');

// Import routes
const authRoutes = require('./routes/authRoutes');
const uploadRoutes = require('./routes/uploadRoutes'); // ✅ ADD THIS

const app = express();

// Initialize Passport (no sessions used)
app.use(passport.initialize());

// Middleware
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/upload', uploadRoutes); // ✅ ADD THIS

const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
