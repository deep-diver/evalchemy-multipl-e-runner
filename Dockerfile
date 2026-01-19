# Local Dockerfile that extends the base evalchemy image with Google GenAI SDK
# Build with: docker build -t deepdiver/evalchemy-multipl-e:0.1-google .
FROM deeepdiver/evalchemy-multipl-e:0.1

# Install the Google GenAI SDK
# This is the new SDK (not the deprecated google-generativeai)
RUN pip install google-genai

# Verify installation
RUN python3 -c "from google import genai; print('google-genai installed successfully')"
