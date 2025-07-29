import sys
sys.path.append('src')

# Use the modern implementation instead
from model.modern_classification import OutClassificationModel

# Test DeBERTa with your framework
try:
    print("🧪 Testing modern DeBERTa implementation...")
    
    # Test with your local DeBERTa v3 large model
    model = OutClassificationModel(
        'deberta-v3', 
        './models/deberta-v3-large',  # Use your local model
        num_labels=4, 
        use_cuda=False,
        args={'value_head': 9}
    )
    print("✅ OutClassificationModel with DeBERTa works!")
    
    # Test a simple prediction
    test_data = [
        ["Test headline", "Test article body", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    ]
    
    predictions, outputs = model.predict(test_data)
    print(f"✅ Prediction test successful! Predicted class: {predictions[0]}")
    print(f"✅ Output shape: {outputs.shape}")
    
    print("\n🎉 All tests passed! You can now use DeBERTa with your project.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()