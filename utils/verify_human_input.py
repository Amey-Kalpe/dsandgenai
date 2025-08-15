import ast
from typing import Optional
from database.database import db


# Helper function for customer identification
def get_customer_id_from_identifier(identifier: str) -> Optional[int]:
    """
    Retrieve Customer ID using an identifier, which can be a customer ID, email, or phone number.

    This function supports three types of identifiers:
    1. Direct customer ID (numeric string)
    2. Phone number (starts with '+')
    3. Email address (contains '@')

    Args:
        identifier (str): The identifier can be customer ID, email, or phone number.

    Returns:
        Optional[int]: The CustomerId if found, otherwise None.
    """
    try:
        # Check if identifier is a direct customer ID (numeric)
        if identifier.isdigit():
            return int(identifier)

        # Check if identifier is a phone number (starts with '+')
        elif identifier[0] == "+":
            query = f"SELECT CustomerId FROM Customer WHERE Phone = '{identifier}';"
            result = db.run(query)
            formatted_result = ast.literal_eval(result)
            if formatted_result:
                return formatted_result[0][0]

        # Check if identifier is an email address (contains '@')
        elif "@" in identifier:
            query = f"SELECT CustomerId FROM Customer WHERE Email = '{identifier}';"
            result = db.run(query)
            formatted_result = ast.literal_eval(result)
            if formatted_result:
                return formatted_result[0][0]

        # Return message if no match found
        return ""
    except Exception:
        return ""
