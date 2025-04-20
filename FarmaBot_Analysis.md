# FarmaBot Project Analysis

## Project Overview
FarmaBot is a pharmaceutical chatbot designed to assist customers with medicine information, store locations, and inventory availability. The system leverages natural language processing to understand user queries and provide relevant responses in both English and Spanish.

## Architecture

### Core Components
1. **Bot Engine** (`core/bot.py`)
   - The central `FarmaBot` class manages the conversation flow
   - Supports both OpenAI and Google Gemini models
   - Handles context tracking for multi-turn conversations
   - Implements various processing functions for different query types (medicine, store, orders)

2. **Services Layer**
   - `DatabaseService` - Handles SQL database connections and queries
   - `MedicineService` - Manages medicine information retrieval and vector search
   - `StoreService` - Handles store-related information and inventory queries

3. **Interface Layer**
   - Gradio-based web interface (`interface/gradio_interface.py`)
   - Bilingual support (English/Spanish)
   - Interactive map for store locations
   - PDF generation for medicine information and orders

### Database Schema
- **Medicines Table** - Stores medicine information including generic names, brand names, uses, side effects
- **Stores Table** - Stores information about pharmacy locations
- **Inventory Tables** - Separate tables for each store location's inventory

### Vector Database
- Chroma vector database stored in the `Medicines/` directory
- Used for semantic search of medicine information
- Each UUID directory represents a collection/partition of the database
- Large SQLite file (`chroma.sqlite3`) contains the vector data

## Key Features

### Medicine Information Retrieval
1. **Multi-strategy search**
   - Direct database lookup
   - Fuzzy matching for similar names
   - Vector similarity search for semantic matching
   - Supports both generic and brand name searches

2. **Medicine information provided**
   - Availability in different store locations
   - Side effects (common and rare)
   - Prescription requirements
   - Similar medicines

### Store Information
1. **Store details**
   - Hours of operation
   - Locations with interactive map
   - Services available

2. **Inventory checking**
   - Real-time inventory levels across stores
   - Store-specific availability

### Additional Functionality
1. **Order processing**
   - Multi-step flow for placing orders
   - PDF quote generation
   - Order history tracking

2. **Language Support**
   - Bilingual responses (English/Spanish)
   - Language-specific formatting

## Technology Stack
- **Language Models**: OpenAI GPT models or Google Gemini models
- **Vector Embeddings**: OpenAI embeddings
- **Vector Database**: Chroma
- **SQL Database**: Microsoft SQL Server
- **Web Interface**: Gradio
- **PDF Generation**: ReportLab
- **Mapping**: Folium (for interactive maps)
- **Language**: Python

## Implementation Details

### NLP Processing Flow
1. User input is processed to determine intent
2. Context tracking for multi-turn conversations
3. Query routing to specialized handlers
4. Response generation in the selected language

### Data Flow
1. User query → Gradio interface
2. Interface → FarmaBot core
3. FarmaBot → Appropriate service (Medicine, Store)
4. Service → Database or vector search
5. Generated response → Interface → User

## Configuration
- Environment variables in `.env.local` control:
  - Database connection
  - API keys (OpenAI or Google)
  - Model selection

## Key Files
- `main.py` - Application entry point
- `config.py` - Configuration settings
- `core/bot.py` - Core chatbot logic
- `services/medicine_service.py` - Medicine-related functionality
- `services/database_service.py` - Database operations
- `services/store_service.py` - Store-related functionality
- `interface/gradio_interface.py` - User interface
- `ChatbotFarmacia.sql` - Database schema definition

## Observations
- The system uses a hybrid search approach combining direct database queries and vector similarity
- Context tracking allows for multi-turn conversations about medicines and orders
- Bilingual support is implemented throughout the application
- The architecture follows a service-oriented design with clear separation of concerns 