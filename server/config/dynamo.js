/**
 * config/dynamo.js
 * This file configures and exports the DynamoDB DocumentClient instance.
 * we can import this wherever we need to interact with DynamoDB.
 */

const AWS = require('aws-sdk');

/**
 * Update AWS configuration:
 *  - region: e.g., us-east-1
 *  - (optionally) accessKeyId, secretAccessKey if not using environment variables or credential files
 */
AWS.config.update({
  region: process.env.AWS_REGION,
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,

});

/**
 * Create a DocumentClient instance for DynamoDB.
 * DocumentClient allows you to work with DynamoDB in a more object-like way.
 */
const dynamoDB = new AWS.DynamoDB.DocumentClient();



// --- DEBUG SNIPPET: Scan "Users" table and log items ---
const params = {
  TableName: 'Users',
};

console.log('DEBUG: Checking DynamoDB region:', AWS.config.region);

dynamoDB.scan(params, (err, data) => {
  if (err) {
    console.error('Scan error:', err);
  } else {
    console.log('Items in "Users" table:', data.Items);
  }
});
// --- END DEBUG SNIPPET ---





module.exports = dynamoDB;
