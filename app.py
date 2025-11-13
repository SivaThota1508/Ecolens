import os
import base64
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import google.genai as genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Setup Flask
app = Flask(__name__)
UPLOAD_FOLDER = '/tmp/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.template_folder = 'templates'  # Set template directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Update configuration for file uploads
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max total upload
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

def load_gemini_key():
    """Load Gemini API key from environment variables"""
    api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment variables")
    return api_key

def initialize_gemini_grounding():
    """Initialize Gemini with Google Search grounding"""
    try:
        gemini_key = load_gemini_key()
        os.environ["GOOGLE_API_KEY"] = gemini_key
        
        client = genai.Client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])
        print("✅ Gemini with Google grounding ready")
        return client, config
    except Exception as e:
        print(f"❌ Gemini grounding setup failed: {e}")
        return None, None

def encode_image(image_path):
    """Encode image in base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def extract_grounding_sources(response):
    """Extract sources from grounding response"""
    sources = []
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                grounding_metadata = candidate.grounding_metadata
                
                if hasattr(grounding_metadata, 'grounding_chunks'):
                    chunks = grounding_metadata.grounding_chunks
                    for chunk in chunks:
                        if hasattr(chunk, 'web'):
                            web = chunk.web
                            source = {
                                'url': str(web.uri) if hasattr(web, 'uri') else '',
                                'title': str(web.title) if hasattr(web, 'title') else 'Unknown Source'
                            }
                            if source['url']:
                                sources.append(source)
    except Exception as e:
        print(f"Error extracting sources: {e}")
    
    return sources

def format_grounding_response_with_ai(raw_response, sources):
    """Use AI to format the grounding response into a neat, comprehensive statement"""
    try:
        gemini_key = load_gemini_key()
        os.environ["GOOGLE_API_KEY"] = gemini_key
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        
        # Prepare sources information for context
        sources_info = ""
        if sources:
            sources_info = "\n\nSources found:\n"
            for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
                sources_info += f"{i}. {source.get('title', 'Unknown')}\n"
        
        formatting_prompt = f"""You are a carbon footprint analyst. Format the following grounding response into a clean, professional statement about the product's carbon footprint. 

Raw grounding response:
{raw_response}

{sources_info}

Instructions:
1. Extract the carbon footprint value (in kg CO2e) if mentioned
2. Create a comprehensive, well-written statement about the product's environmental impact
3. Include methodology details if available (lifecycle assessment, manufacturing, transport, etc.)
4. Mention data source credibility if evident
5. Be factual and professional
6. Return in this exact JSON format:

{{
  "carbon_value": number or null,
  "formatted_statement": "A comprehensive, well-written statement about the carbon footprint findings. Include specific details about methodology, scope, and context. This can be multiple sentences and detailed.",
  "confidence_level": "high/medium/low",
  "methodology_mentioned": "LCA/EPD/company_report/study/unknown"
}}

Focus on creating a detailed, informative statement that professionals would find useful. No character limits."""

        message = HumanMessage(content=[{"type": "text", "text": formatting_prompt}])
        response = llm([message])
        
        return parse_ai_formatted_response(response.content)
        
    except Exception as e:
        print(f"Error formatting response with AI: {e}")
        # Fallback to original parsing
        carbon_value, statement = parse_carbon_footprint_response_fallback(raw_response)
        return carbon_value, statement, "low", "unknown"

def parse_ai_formatted_response(ai_response):
    """Parse the AI-formatted response"""
    try:
        # Clean and extract JSON
        response_text = ai_response.strip()
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            parsed_data = json.loads(json_text)
            
            return (
                parsed_data.get('carbon_value'),
                parsed_data.get('formatted_statement', 'No carbon footprint information found.'),
                parsed_data.get('confidence_level', 'low'),
                parsed_data.get('methodology_mentioned', 'unknown')
            )
        else:
            raise ValueError("No valid JSON in AI response")
            
    except Exception as e:
        print(f"Error parsing AI formatted response: {e}")
        # Fallback
        return None, "Unable to format carbon footprint information.", "low", "unknown"

def parse_carbon_footprint_response_fallback(response_text):
    """Fallback method for parsing carbon footprint (original logic)"""
    import re
    
    # Look for carbon footprint patterns
    patterns = [
        r'(\d+\.?\d*)\s*kg\s*CO2e?\s*/?.*',
        r'(\d+\.?\d*)\s*kg\s*CO2\s*equivalent',
        r'carbon footprint.*?(\d+\.?\d*)\s*kg',
        r'(\d+\.?\d*)\s*kg.*CO2',
        r'emissions?.*?(\d+\.?\d*)\s*kg'
    ]
    
    carbon_value = None
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                if 0 < value < 10000:  # Reasonable range
                    carbon_value = value
                    break
            except (ValueError, IndexError):
                continue
    
    # Extract a relevant statement
    sentences = response_text.split('.')
    statement = "No specific carbon footprint statement found in search results."
    
    for sentence in sentences:
        if any(word in sentence.lower() for word in ['carbon', 'footprint', 'emission', 'co2']):
            statement = sentence.strip() + '.'
            break
    
    return carbon_value, statement

def analyze_product_image(image_path):
    """Analyze product from image using Gemini"""
    try:
        gemini_key = load_gemini_key()
        os.environ["GOOGLE_API_KEY"] = gemini_key
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        base64_image = encode_image(image_path)

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": """Analyze this product image and provide basic information in JSON format only.

Return ONLY valid JSON with no additional text:

{
  "brand": "Brand name if visible or identifiable",
  "product_name": "Specific product name/model if identifiable", 
  "product_type": "Category (e.g., smartphone, laptop, shoes, bottle)",
}

Be specific about the brand and product type. If you see text on the product, read it carefully.
IMPORTANT: Respond with ONLY the JSON object."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        )
        
        response = llm([message])
        return parse_product_response(response.content)
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return None

def parse_product_response(response_text):
    """Parse product details from AI response"""
    try:
        # Clean the response to extract JSON
        response_text = response_text.strip()
        
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # Find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = response_text[json_start:json_end]
            parsed_data = json.loads(json_text)
            
            return {
                'brand': str(parsed_data.get('brand', 'Unknown')).strip() or 'Unknown',
                'product_name': str(parsed_data.get('product_name', 'Unknown')).strip() or 'Unknown',
                'product_type': str(parsed_data.get('product_type', 'Unknown')).strip() or 'Unknown'
            }
        else:
            raise ValueError("No valid JSON found")
            
    except Exception as e:
        print(f"Error parsing product response: {e}")
        return {
            'brand': 'Unknown',
            'product_name': 'Unknown',
            'product_type': 'Unknown'
        }

def get_carbon_footprint_benchmark(product_brand, product_name, product_type):
    """Get carbon footprint benchmark using Google grounding"""
    try:
        client, config = initialize_gemini_grounding()
        if not client:
            return None, "Grounding not available", [], "low", "unknown"
        
        # Search prompt for carbon footprint
        prompt = f"Find carbon footprint value for: {product_brand} {product_name} {product_type}. Search environmental product declarations, LCA studies, and sustainability reports for specific carbon footprint data in kg CO2e."

        print(f"🔍 Searching carbon footprint for: {product_brand} {product_name}")
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=config,
        )
        
        # Extract sources
        sources = extract_grounding_sources(response)
        
        # Format response using AI for better presentation
        carbon_value, formatted_statement, confidence, methodology = format_grounding_response_with_ai(response.text, sources)
        
        print(f"✅ Found carbon footprint: {carbon_value} kg CO2e")
        print(f"📊 Sources found: {len(sources)}")
        print(f"🎯 Confidence: {confidence}")
        
        return carbon_value, formatted_statement, sources, confidence, methodology
        
    except Exception as e:
        print(f"❌ Benchmark search failed: {e}")
        return None, f"Search failed: {str(e)}", [], "low", "unknown"

def validate_image_file(file):
    """Validate uploaded image file"""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    # Check file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return False, f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
    
    # Check file size (10MB per file)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset to beginning
    
    if file_size > 10 * 1024 * 1024:
        return False, "File size exceeds 10MB limit"
    
    return True, "Valid file"

# Web Routes
@app.route('/')
def home():
    """Landing page"""
    return render_template('landing.html')

@app.route('/single-analysis')
def single_analysis():
    """Single image analysis page"""
    return render_template('single_analysis.html')

@app.route('/manual-entry')
def manual_entry():
    """Manual product entry page"""
    return render_template('manual_entry.html')

@app.route('/batch-analysis')
def batch_analysis():
    """Batch analysis page"""
    return render_template('batch_analysis.html')

# API Routes
@app.route('/api/analyze', methods=['POST'])
def analyze_product():
    """Main API endpoint: analyze image and return product with carbon footprint"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400

        image = request.files['image']
        
        # Validate file
        is_valid, message = validate_image_file(image)
        if not is_valid:
            return jsonify({'success': False, 'error': message}), 400

        # Save image
        filename = secure_filename(image.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        image.save(filepath)

        # Step 1: Analyze product from image
        print("🔍 Analyzing product image...")
        product_info = analyze_product_image(filepath)
        
        if not product_info:
            # Clean up file
            try:
                os.remove(filepath)
            except:
                pass
            return jsonify({'success': False, 'error': 'Failed to analyze product from image'}), 500

        # Step 2: Get carbon footprint benchmark
        print("🌍 Searching carbon footprint data...")
        carbon_value, formatted_statement, sources, confidence, methodology = get_carbon_footprint_benchmark(
            product_info['brand'],
            product_info['product_name'], 
            product_info['product_type']
        )

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        # Prepare response
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'product': {
                'brand': product_info['brand'],
                'product_name': product_info['product_name'],
                'product_type': product_info['product_type']
            },
            'carbon_footprint': {
                'value_kg_co2e': carbon_value,
                'statement': formatted_statement,
                'confidence_level': confidence,
                'methodology': methodology,
                'sources': sources[:10] if sources else []  # Limit to 10 sources
            }
        }

        return jsonify(result), 200

    except Exception as e:
        # Clean up on error
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass
        
        print(f"Error in analyze_product: {e}")
        return jsonify({
            'success': False,
            'error': 'Analysis failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/analyze/manual', methods=['POST'])
def analyze_manual():
    """Manual product entry endpoint"""
    try:
        # Get data from request
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Validate required fields
        required_fields = ['brand', 'product_name', 'product_type']
        for field in required_fields:
            if not data.get(field) or not data[field].strip():
                return jsonify({
                    'success': False, 
                    'error': f'Missing or empty required field: {field}'
                }), 400
        
        product_info = {
            'brand': data['brand'].strip(),
            'product_name': data['product_name'].strip(),
            'product_type': data['product_type'].strip()
        }
        
        print(f"🔍 Manual analysis for: {product_info['brand']} {product_info['product_name']}")
        
        # Get carbon footprint benchmark
        print("🌍 Searching carbon footprint data...")
        carbon_value, formatted_statement, sources, confidence, methodology = get_carbon_footprint_benchmark(
            product_info['brand'],
            product_info['product_name'], 
            product_info['product_type']
        )

        # Prepare response
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'input_method': 'manual',
            'product': product_info,
            'carbon_footprint': {
                'value_kg_co2e': carbon_value,
                'statement': formatted_statement,
                'confidence_level': confidence,
                'methodology': methodology,
                'sources': sources[:10] if sources else []
            }
        }

        return jsonify(result), 200

    except Exception as e:
        print(f"Error in analyze_manual: {e}")
        return jsonify({
            'success': False,
            'error': 'Manual analysis failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'input_method': 'manual'
        }), 500

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    """Batch image analysis endpoint"""
    try:
        uploaded_files = request.files.getlist('images')
        
        if not uploaded_files or len(uploaded_files) == 0:
            return jsonify({'success': False, 'error': 'No images provided'}), 400

        # Filter valid image files and limit to 10
        valid_files = []
        errors = []
        
        for i, file in enumerate(uploaded_files[:10]):  # Limit to 10
            is_valid, message = validate_image_file(file)
            if is_valid:
                valid_files.append(file)
            else:
                errors.append(f"File {i+1} ({file.filename}): {message}")

        if not valid_files:
            return jsonify({
                'success': False, 
                'error': 'No valid image files found',
                'validation_errors': errors
            }), 400

        print(f"🚀 Starting batch analysis: {len(valid_files)} valid images")
        
        batch_start_time = datetime.now()
        results = []
        successful_count = 0
        failed_count = 0

        # Process each image
        for index, image_file in enumerate(valid_files):
            try:
                print(f"📷 Processing image {index + 1}/{len(valid_files)}: {image_file.filename}")
                
                # Save image temporarily
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = secure_filename(f"batch_{index}_{timestamp}_{image_file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image_file.save(filepath)
                
                # Analyze product from image
                product_info = analyze_product_image(filepath)
                
                if product_info:
                    # Get carbon footprint
                    carbon_value, formatted_statement, sources, confidence, methodology = get_carbon_footprint_benchmark(
                        product_info['brand'],
                        product_info['product_name'], 
                        product_info['product_type']
                    )
                    
                    result = {
                        'success': True,
                        'index': index,
                        'filename': image_file.filename,
                        'product': product_info,
                        'carbon_footprint': {
                            'value_kg_co2e': carbon_value,
                            'statement': formatted_statement,
                            'confidence_level': confidence,
                            'methodology': methodology,
                            'sources': sources[:5] if sources else []  # Fewer sources for batch
                        }
                    }
                    successful_count += 1
                else:
                    result = {
                        'success': False,
                        'index': index,
                        'filename': image_file.filename,
                        'error': 'Failed to analyze product from image'
                    }
                    failed_count += 1
                
                results.append(result)
                
                # Clean up file
                try:
                    os.remove(filepath)
                except:
                    pass
                        
            except Exception as e:
                print(f"Error processing {image_file.filename}: {e}")
                results.append({
                    'success': False,
                    'index': index,
                    'filename': image_file.filename,
                    'error': str(e)
                })
                failed_count += 1

        # Calculate batch summary
        batch_processing_time = (datetime.now() - batch_start_time).total_seconds()

        batch_result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'input_method': 'batch',
            'summary': {
                'total_images': len(valid_files),
                'successful_analyses': successful_count,
                'failed_analyses': failed_count,
                'processing_time_seconds': round(batch_processing_time, 2),
                'average_time_per_image': round(batch_processing_time / len(valid_files), 2) if valid_files else 0
            },
            'results': results,
            'validation_errors': errors if errors else []
        }

        print(f"✅ Batch analysis completed: {successful_count}/{len(valid_files)} successful")
        
        return jsonify(batch_result), 200

    except Exception as e:
        print(f"Error in analyze_batch: {e}")
        return jsonify({
            'success': False,
            'error': 'Batch analysis failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat(),
            'input_method': 'batch'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test Gemini connection
        gemini_key = load_gemini_key()
        gemini_ready = bool(gemini_key)
        
        # Test grounding
        client, config = initialize_gemini_grounding()
        grounding_ready = bool(client and config)
        
        return jsonify({
            'status': 'healthy' if (gemini_ready and grounding_ready) else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'services': {
                'gemini_api': 'ready' if gemini_ready else 'not_configured',
                'google_grounding': 'ready' if grounding_ready else 'not_configured'
            },
            'endpoints': {
                'image_analysis': '/api/analyze (POST with image)',
                'manual_analysis': '/api/analyze/manual (POST with JSON)',
                'batch_analysis': '/api/analyze/batch (POST with multiple images)',
                'health': '/api/health (GET)'
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/founders')
def founders():
    """Founders page"""
    return render_template('founders.html')
# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors - redirect to home for SPA-like behavior"""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'API endpoint not found',
            'message': 'The requested API endpoint was not found.'
        }), 404
    else:
        # Redirect unknown routes to home page
        return render_template('landing.html')

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    if request.path.startswith('/api/'):
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred. Please try again.'
        }), 500
    else:
        return render_template('landing.html')

@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors"""
    return jsonify({
        'success': False,
        'error': 'File too large',
        'message': 'Uploaded file exceeds the maximum size limit of 10MB per file.'
    }), 413

# Add CORS headers for frontend integration
@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == '__main__':
    print("🚀 Starting EcoLens - AI-Powered Carbon Footprint Analysis...")
    print("📋 Required environment variables:")
    print("   - GEMINI_API_KEY (or GOOGLE_API_KEY)")
    print("🌐 Web Interface:")
    print("   - http://localhost:5000/ - Landing page")
    print("   - http://localhost:5000/single-analysis - Single image analysis")
    print("   - http://localhost:5000/manual-entry - Manual product entry") 
    print("   - http://localhost:5000/batch-analysis - Batch analysis (max 10 images)")
    print("🔌 API Endpoints:")
    print("   - POST /api/analyze - Single image analysis")
    print("   - POST /api/analyze/manual - Manual product entry")
    print("   - POST /api/analyze/batch - Batch analysis")
    print("   - GET /api/health - Health check")
    print("📁 Make sure to create templates/ directory with HTML files")
    
    app.run(host='0.0.0.0', port=5000, debug=True)