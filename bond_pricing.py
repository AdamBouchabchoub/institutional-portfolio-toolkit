import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def bond_price(face_value, coupon_rate, ytm, years_to_maturity, frequency=2):
    """
    Calculate Bond price.

    Parameters:
    -----------
    face_value : float
        Face Value of bond (100 or 1000)
    coupon_rate : float
       Annual coupon rate (e.g 0.05 for 5%)
    ytm : float
       Yield to maturity (e.g 0.05 for 5%)
    years_to_maturity: float
       Number of years until maturity
    frequency : int
       Number of payement per year(2=semi-annual, 1=annual)

     Returns:
     --------
     float : Bond price
    """
    annual_coupon = face_value * coupon_rate
    coupon_per_period = annual_coupon / frequency
    periods = int(years_to_maturity*frequency)
    n = np.arange(1, periods + 1)
    yield_per_period = ytm / frequency
    
    pv_coupons = coupon_per_period / (1 + yield_per_period)**n
    pv_face = face_value / (1+yield_per_period) ** periods
    price = pv_coupons.sum() + pv_face

    return price

def accrued_interest(face_value, coupon_rate, days_since_last_coupon, days_in_coupon_period, frequency=2):
    """
    Calculate accrued interest.

    Parametrs:
    ----------
    face_value : float
        Face Value of bond (100 or 1000)
    coupon_rate: float
        Annual coupon rate (e.g 0.05 for 5%)
    days_since_last_coupon: int
        Days since last coupon payment
    days_in_coupon_period: int
        Days between two coupon payment (Typically 180 for semi-annual)
    frequency: int
        Number of payements per year(2=semi-annual, 1=annual)

    Returns:
    --------
    float: Accrued interest   
    """
    annual_coupon = face_value * coupon_rate
    coupon_per_period = annual_coupon / frequency
    accrued_interest = (days_since_last_coupon / days_in_coupon_period) * coupon_per_period

    return accrued_interest
    

def dirty_price(clean_price, accrued_interest):
    """
    Calculate dirty price.
    Dirty price = clean price + accrued interest
    
    Parameters:
    -----------
    clean_price : float
        Quoted bond price (without accrued interest)
    accrued_interest : float
        Interest accrued since last coupon payment
        
    Returns:
    --------
    float : Dirty price (actual price paid)
    """
    dirty = clean_price + accrued_interest

    return dirty
    

def macaulay_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency=2):
    """
    Calculate Macaulay Duration.
    
    Parameters:
    ----------
    face_value : float
        Face Value of bond (100 or 1000)
    coupon_rate : float
       Annual coupon rate (e.g 0.05 for 5%)
    ytm : float
       Yield to maturity (e.g 0.05 for 5%)
    years_to_maturity: float
       Number of years until maturity
    frequency : int
       Number of payement per year(2=semi-annual, 1=annual)
       
    Returns:
    -------
    float : Duration in years
    """
    annual_coupon = face_value * coupon_rate
    coupon_per_period = annual_coupon / frequency
    periods = int(years_to_maturity * frequency)
    n = np.arange(1, periods+1)
    yield_per_period = ytm / frequency

    pv_coupon = coupon_per_period / (1 + yield_per_period)**n
    pv_face = face_value / (1+ yield_per_period)**periods
    price = pv_face + pv_coupon.sum()
    
    pv_weighted_coupon = (n * coupon_per_period) / (1 + yield_per_period)**n
    pv_weighted_face = (periods * face_value) / (1+ yield_per_period)**periods
    
    mac_dur_period = (pv_weighted_coupon.sum() + pv_weighted_face) / price
    mac_dur_annual =  mac_dur_period / frequency

    return mac_dur_annual

def modified_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency=2):
    """
    Calculate Modified Duration.
    (Bond price change for 1% change in yield)
    
    Parameters:
    ----------
    face_value : float
        Face Value of bond (100 or 1000)
    coupon_rate : float
       Annual coupon rate (e.g 0.05 for 5%)
    ytm : float
       Yield to maturity (e.g 0.05 for 5%)
    years_to_maturity: float
       Number of years until maturity
    frequency : int
       Number of payement per year(2=semi-annual, 1=annual)
       
    Returns:
    -------
    float : Modified Duration
    """
    mac_dur = macaulay_duration(face_value, coupon_rate, ytm, years_to_maturity, frequency)
    modified_duration = mac_dur / (1 + ytm/frequency)

    return modified_duration

def convexity(face_value, coupon_rate, ytm, years_to_maturity, frequency=2):
    """
    Calculate Convexity.
    
    Convexity measures the curvature of the price-yield relationship.

    Parameters:
    -----------
    face_value : float
        Face Value of bond (100 or 1000)
    coupon_rate : float
       Annual coupon rate (e.g 0.05 for 5%)
    ytm : float
       Yield to maturity (e.g 0.05 for 5%)
    years_to_maturity: float
       Number of years until maturity
    frequency : int
       Number of payement per year(2=semi-annual, 1=annual)
       
    Returns:
    --------
    float : Convexity
    
    """
    annual_coupon = face_value * coupon_rate
    coupon_per_period = annual_coupon / frequency
    periods = int(years_to_maturity * frequency)
    n = np.arange(1, periods+1)
    yield_per_period = ytm / frequency

    pv_coupon = coupon_per_period / (1 + yield_per_period)**n
    pv_face = face_value / (1+ yield_per_period)**periods
    price = pv_face + pv_coupon.sum()

    pv_weighted_coupon = (n * (n+1)* coupon_per_period) / (1 + yield_per_period)**n
    pv_weighted_face = (periods* (periods + 1) * face_value) / (1+ yield_per_period)**periods

    conv = (pv_weighted_coupon.sum() + pv_weighted_face) / (price * (1 + yield_per_period)**2)

    return conv
    
def price_change_estimate(modified_duration, convexity, yield_change, current_price):
    """
    Estimate bond price change using duration-convexity approximation.
    
    Parameters:
    -----------
    modified_duration : float
        Modified duration of the bond
    convexity : float
        Convexity of the bond
    yield_change : float
        Change in yield (e.g., 0.01 for +1%, -0.005 for -0.5%)
    current_price : float
        Current bond price
        
    Returns:
    --------
    float : Estimated new price
    """
    duration_effect = -modified_duration * yield_change
    convexity_effect =  0.5 * convexity * (yield_change ** 2)
    pct_change = duration_effect + convexity_effect
    new_price = current_price * (1 + pct_change)
    
    return new_price

def compare_bonds(bonds_dict):
    """
    Compare multiple bonds et affiche tableau avec metrics.
    
    Parameters:
    -----------
    bonds_dict : dict
        Format: {'Bond Name': {'face_value': 100, 'coupon_rate': 0.05, ...}}
    
    Returns:
    --------
    pd.DataFrame : Compararison DataFrame
    """
    results = []
    for name, params in bonds_dict.items(): 
        price = bond_price(**params)
        mac_dur = macaulay_duration(**params)
        mod_dur = modified_duration(**params)
        conv = convexity(**params)
        
        results.append({'Bond': name,
                        'Price': price,
                        'Coupon': params['coupon_rate'],
                        'YTM': params['ytm'],  
                        'Maturity': params['years_to_maturity'],
                        'Mac Dur': mac_dur,
                        'Mod Dur': mod_dur,
                        'Convexity': conv})
    df = pd.DataFrame(results)
    return df

def plot_price_yield_curve(face_value=100, coupon_rate=0.05, years_to_maturity=10, current_ytm=None):

    if current_ytm is None:
        current_ytm = coupon_rate
        
    ytm_range = np.linspace(0.01, 0.12, 100)
    prices = [bond_price(face_value, coupon_rate, y, years_to_maturity) for y in ytm_range]
    current_price = bond_price(face_value, coupon_rate, current_ytm, years_to_maturity)
    mod_dur = modified_duration(face_value, coupon_rate, current_ytm, years_to_maturity)
    conv = convexity(face_value, coupon_rate, current_ytm, years_to_maturity)
    duration_prices = [current_price * (1-mod_dur * (y - current_ytm)) for y in ytm_range]
    conv_prices = [current_price * (1 - mod_dur * (y - current_ytm) + 0.5 * conv * (y - current_ytm)**2) for y in ytm_range]

    plt.figure(figsize=(16,8))
    plt.plot(ytm_range * 100, prices, label='Bond Prices Curve')
    plt.plot(ytm_range*100, duration_prices, '--', label='Duration Approx')
    plt.plot(ytm_range*100, conv_prices, ':', label='Duration + Convexity')
    plt.scatter([current_ytm * 100], [current_price], s=80, c='red', zorder=5)
    plt.xlabel('Yield (%)')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BOND ANALYTICS TEST")
    print("="*60)

    face = 100
    coupon = 0.05
    ytm = 0.05
    years = 10

    price = bond_price(face, coupon, ytm, years)
    mac_dur = macaulay_duration(face, coupon, ytm, years) 
    mod_dur = modified_duration(face, coupon, ytm, years)
    conv = convexity(face, coupon, ytm, years)

    print(f"Bond: ${face}, {coupon: .1%} coupon, {ytm: .1%} YTM, {years} years")
    print(f"Price: ${price: .2f}")
    print(f"Macaulay Duration: {mac_dur:.2f} years")
    print(f"Modified Duration: {mod_dur:.2f}")
    print(f"Convexity: {conv:.2f}")

    print("\n" + "="*60)
    print("YIELD CHANGE SCENARIO: +100bps")
    print("="*60)

    yield_change = 0.01
    new_price_estimate = price_change_estimate(mod_dur, conv, yield_change, price)
    new_price_actual = bond_price(face, coupon, ytm + yield_change, years)
    error = abs(new_price_estimate - new_price_actual)

    print(f"Current Price: ${price: .2f}")
    print(f"Estimated new price: ${new_price_estimate: .2f}")
    print(f"Actual new price: ${new_price_actual: .2f}")
    print(f"Approximation error: ${error:.4f} ({error/new_price_actual:.2%})")

    print("\n" + "="*60)
    print("BOND COMPARISON TEST")
    print("="*60)

    bonds = {'Par Bond': {'face_value': 100, 'coupon_rate': 0.05, 'ytm': 0.05, 'years_to_maturity': 10},
            'Premium Bond': {'face_value': 100, 'coupon_rate': 0.07, 'ytm': 0.05, 'years_to_maturity': 10}, 
            'Discount Bond': {'face_value': 100, 'coupon_rate': 0.03, 'ytm': 0.05, 'years_to_maturity': 10}, 
            'Zero-Coupon': {'face_value': 100, 'coupon_rate': 0.00, 'ytm': 0.05, 'years_to_maturity': 10}, 
            'Short-Term': {'face_value': 100, 'coupon_rate': 0.05, 'ytm': 0.05, 'years_to_maturity': 2}}
    df = compare_bonds(bonds)
    print(df.to_string(index=False))

    print("\n" + "="*60)
    print("PRICE-YIELD CURVE VISUALIZATION")
    print("=" * 60)
    print("Generating plot...\n")

    plot_price_yield_curve(face_value=100, coupon_rate=0.05, years_to_maturity=10)

    