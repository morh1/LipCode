/**
 * services/authService.js
 * This file contains the core "business logic" related to authentication.
 * Each function can interact with the database and return data or throw errors.
 */

const dynamoDB = require('../config/dynamo');  // DynamoDB config
const TABLE_NAME = 'Users';                    // The DynamoDB table name for user records

/**
 * Checks if a user exists in the database by email.
 * @param {string} email - The user's email address.
 * @returns {Promise<boolean>} - True if the user exists, false otherwise.
 */
async function checkUserExists(email) {
  const params = {
    TableName: TABLE_NAME,
    Key: { email },     // Using email as the primary key
  };
  const result = await dynamoDB.get(params).promise();
  // result.Item will be defined if the email was found
  return !!result.Item;
}

/**
 * Create a user record in the DB. No 'activated' field needed,
 * because if they're in the DB, they're considered activated.
 * @param {string} email
 */
async function createUser(email) {
  const params = {
    TableName: TABLE_NAME,
    Item: {
      email,
      // Add any other fields you want to store, like createdAt, name, etc.
    },
  };
  await dynamoDB.put(params).promise();
}

module.exports = {
  checkUserExists,
  createUser,
};

//You can add more complicated logic here, such as sending activation emails, hashing user data, etc