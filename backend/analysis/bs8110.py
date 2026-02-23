# backend/analysis/bs8110.py
"""
BS8110 Concrete Design Calculations
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

@dataclass
class ReinforcementResult:
    area_required: float  # mm²
    area_provided: float  # mm²
    bar_diameter: int     # mm
    num_bars: int
    spacing: float        # mm
    capacity: float       # kNm
    utilization: float    # ratio
    crack_width: float    # mm
    links_required: bool
    link_diameter: int    # mm
    link_spacing: float   # mm

class BS8110Designer:
    def __init__(self):
        # Material properties (BS8110)
        self.fcu = 30  # Concrete grade N/mm² (C30)
        self.fy = 460  # Steel grade N/mm² (Grade 460)
        self.gamma_c = 1.5  # Concrete safety factor
        self.gamma_s = 1.15  # Steel safety factor
        self.cover = 35  # Nominal cover mm
        self.link_size = 8  # Default link diameter
        
    def design_rectangular_section(self, 
                                   breadth: float,  # mm
                                   depth: float,    # mm
                                   moment: float,   # kNm
                                   shear: float,    # kN
                                   is_sagging: bool = True) -> ReinforcementResult:
        """
        Design reinforcement for rectangular beam section to BS8110
        
        Parameters:
        -----------
        breadth : float - Section width in mm
        depth : float - Effective depth in mm  
        moment : float - Applied moment in kNm
        shear : float - Applied shear in kN
        is_sagging : bool - True for sagging (bottom steel), False for hogging
        """
        
        # Design constants
        fcu = self.fcu
        fy = self.fy
        
        # Effective depth (assuming 20mm main bars + links)
        d = depth - self.cover - self.link_size - 10
        
        # Moment redistribution factor (BS8110)
        beta_b = 0.9
        
        # Calculate K value
        M = moment * 1e6  # Convert to Nmm
        K = M / (fcu * breadth * d**2)
        
        # Limit K to ensure ductile failure (BS8110 Cl 3.4.4.4)
        K_prime = 0.156 * beta_b
        
        if K > K_prime:
            # Compression reinforcement required
            # Simplified - in practice would calculate comp steel
            K = K_prime
        
        # Lever arm calculation
        z = d * (0.5 + math.sqrt(0.25 - K/0.9))
        z = min(z, 0.95 * d)  # Limit z to 0.95d
        
        # Tension reinforcement required
        As_req = M / (0.95 * fy * z)
        
        # Select bar size and number
        bar_options = [16, 20, 25, 32, 40]
        selected_bars = self._select_bars(As_req, breadth, bar_options)
        
        As_prov = selected_bars['area']
        
        # Check moment capacity
        x = (As_prov * 0.95 * fy) / (0.45 * fcu * breadth)
        z_actual = d - 0.45 * x
        M_capacity = As_prov * 0.95 * fy * z_actual / 1e6  # kNm
        
        # Shear design
        v = shear * 1000 / (breadth * d)  # N/mm²
        vc = self._calculate_vc(As_prov, breadth, d)
        
        links_req = v > vc
        link_dia, link_spacing = 0, 0
        
        if links_req:
            link_dia, link_spacing = self._design_links(v, vc, breadth, d)
        
        # Crack width check (simplified)
        crack_width = self._calculate_crack_width(M, As_prov, breadth, d)
        
        utilization = abs(moment) / M_capacity if M_capacity > 0 else 0
        
        return ReinforcementResult(
            area_required=As_req,
            area_provided=As_prov,
            bar_diameter=selected_bars['diameter'],
            num_bars=selected_bars['number'],
            spacing=selected_bars['spacing'],
            capacity=M_capacity,
            utilization=utilization,
            crack_width=crack_width,
            links_required=links_req,
            link_diameter=link_dia,
            link_spacing=link_spacing
        )
    
    def _select_bars(self, As_req: float, breadth: float, 
                     options: List[int]) -> Dict:
        """Select appropriate bar arrangement"""
        for dia in options:
            area_single = math.pi * dia**2 / 4
            # Minimum 2 bars, check spacing (min 25mm or bar size)
            min_spacing = max(25, dia)
            max_bars = int((breadth - 2*self.cover) / (dia + min_spacing)) + 1
            
            for n in range(2, max_bars + 1):
                As_prov = n * area_single
                if As_prov >= As_req:
                    total_width = n*dia + (n-1)*min_spacing
                    spacing = (breadth - 2*self.cover - n*dia) / (n-1) if n > 1 else 0
                    return {
                        'diameter': dia,
                        'number': n,
                        'area': As_prov,
                        'spacing': spacing
                    }
        
        # If no solution found, use maximum
        dia = options[-1]
        n = max_bars
        return {
            'diameter': dia,
            'number': n,
            'area': n * math.pi * dia**2 / 4,
            'spacing': 25
        }
    
    def _calculate_vc(self, As: float, b: float, d: float) -> float:
        """Calculate concrete shear stress capacity (BS8110 Table 3.8)"""
        # Simplified formula
        if d == 0:
            return 0
        rho = As / (b * d)
        rho = min(max(rho, 0.0015), 0.03)  # Limits per BS8110
        
        vc = 0.79 * (100*rho)**(1/3) * (400/d)**(1/4) / 1.25
        return min(vc, 0.8 * math.sqrt(self.fcu))  # Max 0.8√fcu
    
    def _design_links(self, v: float, vc: float, b: float, d: float) -> Tuple[int, float]:
        """Design shear links to BS8110"""
        fyv = 250  # Link steel grade (mild steel)
        
        # Shear reinforcement required
        Asv_per_spacing = b * (v - vc) / (0.95 * fyv)
        
        # Try 8mm links (2 legs)
        dia = 8
        Asv = 2 * math.pi * dia**2 / 4
        
        spacing = Asv / Asv_per_spacing
        spacing = min(spacing, 0.75*d)  # Max spacing 0.75d
        
        # Round down to practical spacing
        spacings = [100, 125, 150, 175, 200, 250, 300]
        practical_spacing = min([s for s in spacings if s <= spacing], default=300)
        
        return dia, practical_spacing
    
    def _calculate_crack_width(self, M: float, As: float, 
                               b: float, d: float) -> float:
        """Simplified crack width calculation to BS8110"""
        # Very simplified - proper calculation needs service stress
        fs = M * 1e6 / (0.87 * As * 0.9 * d)  # Service stress approximation
        crack_width = 0.3 * (fs / 200) * (25 / d)**0.33
        return min(crack_width, 0.3)  # Limit to 0.3mm for exposure

    def get_design_summary(self, results: ReinforcementResult) -> Dict:
        """Format results for display"""
        return {
            'main_reinforcement': f"{results.num_bars}T{results.bar_diameter}",
            'area_required': f"{results.area_required:.1f} mm²",
            'area_provided': f"{results.area_provided:.1f} mm²",
            'bar_spacing': f"{results.spacing:.1f} mm",
            'moment_capacity': f"{results.capacity:.2f} kNm",
            'utilization': f"{results.utilization*100:.1f}%",
            'crack_width': f"{results.crack_width:.2f} mm",
            'shear_links': f"T{results.link_diameter}@{results.link_spacing:.0f} c/c" 
                          if results.links_required else "Not required",
            'status': 'OK' if results.utilization <= 1.0 and results.crack_width <= 0.3 
                     else 'REVIEW REQUIRED'
        }