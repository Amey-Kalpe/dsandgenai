# Helper function to format user memory data for LLM prompts
def format_user_memory(user_data):
    """Formats music preferences from users, if available."""
    # Access the 'memory' key which holds the UserProfile object
    profile = user_data["memory"]
    result = ""
    # Check if music_preferences attribute exists and is not empty
    if hasattr(profile, "music_preferences") and profile.music_preferences:
        result += f"Music Preferences: {', '.join(profile.music_preferences)}"
    return result.strip()
