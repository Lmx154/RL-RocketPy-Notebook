"""
Quick test to verify rocket_config integration works correctly.
Run this after installing rocketpy to ensure everything is set up properly.
"""

import sys
import os

# Add notebooks/v-10 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'notebooks', 'v-10'))

try:
    import rocket_config
    print("✓ rocket_config module imported successfully")
    
    # Test configuration constants
    print(f"\n📍 Launch Site:")
    print(f"   Latitude: {rocket_config.LAUNCH_LATITUDE}°N")
    print(f"   Longitude: {rocket_config.LAUNCH_LONGITUDE}°W")
    print(f"   Elevation: {rocket_config.LAUNCH_ELEVATION} m")
    
    print(f"\n🚀 Rocket Dimensions:")
    print(f"   Body Radius: {rocket_config.ROCKET_BODY_RADIUS} m")
    print(f"   Nose Cone Length: {rocket_config.NOSECONE_LENGTH} m")
    print(f"   Fin Count: {rocket_config.FIN_COUNT}")
    
    print(f"\n⚖️  Mass Properties:")
    print(f"   Dry Mass: {rocket_config.ROCKET_DRY_MASS} kg")
    print(f"   Inertia (pitch): {rocket_config.ROCKET_INERTIA_I} kg⋅m²")
    
    print("\n" + "="*60)
    print("Configuration module is working correctly!")
    print("="*60)
    
    # Try creating rocket components (requires rocketpy)
    try:
        print("\nAttempting to create rocket components...")
        rocket = rocket_config.create_rocket(include_parachutes=False, drag_data_path='data')
        print(f"✓ Rocket created successfully!")
        print(f"   Mass: {rocket.mass:.3f} kg")
        print(f"   Radius: {rocket.radius:.4f} m")
        print(f"   Surfaces: {len(rocket.aerodynamic_surfaces)}")
        print("\n🎉 All tests passed! Configuration is fully functional.")
    except ImportError as e:
        print(f"\n⚠️  RocketPy not installed: {e}")
        print("Install RocketPy to test full rocket creation.")
        print("Configuration constants are working correctly though!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
