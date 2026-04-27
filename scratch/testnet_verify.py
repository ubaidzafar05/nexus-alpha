
import asyncio
import os
from pathlib import Path
import ccxt.async_support as ccxt
from dotenv import load_dotenv

async def verify_testnet():
    print("--- NEXUS-ULTRA v9 Testnet Connectivity Check ---")
    
    # Load env
    env_path = Path(".env")
    if not env_path.exists():
        print("Error: .env file not found")
        return
    
    load_dotenv(dotenv_path=env_path)
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    if not api_key or not api_secret:
        print("Error: BINANCE_API_KEY or BINANCE_API_SECRET missing in .env")
        return
    
    print(f"Key: {api_key[:6]}...{api_key[-4:]}")
    print(f"Testnet Mode: {testnet}")
    
    exchange = ccxt.binance({
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })
    
    if testnet:
        exchange.set_sandbox_mode(True)
        
    try:
        print("Connecting to Binance (Spot)...")
        # Ensure we only care about spot
        exchange.options['defaultType'] = 'spot'
        
        # Test basic connectivity before loading markets
        server_time = await exchange.fetch_time()
        print(f"✅ Basic Connectivity Verified. Server Time: {server_time}")
        
        balance = await exchange.fetch_balance()
        print("✅ Auth/Balance Successful!")
        
        totals = {k: v for k, v in balance['total'].items() if v > 0}
        print(f"Non-zero Balances: {totals}")
        
        usdt = totals.get('USDT', 0.0)
        if usdt < 10.0:
            print(f"⚠️ Warning: USDT balance ({usdt}) is below the typical $10 minimum order size on Binance.")
            
        # Try fetching markets to ensure full auth
        await exchange.load_markets()
        print("✅ Market Data Loaded Successfully")
        
    except Exception as e:
        import traceback
        print(f"❌ Connection Failed: {str(e)}")
        traceback.print_exc()
    finally:
        await exchange.close()

if __name__ == "__main__":
    asyncio.run(verify_testnet())
