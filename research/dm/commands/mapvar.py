import numpy as np
import matplotlib.pyplot as plt
from commands.command import Command
from variables import VARIABLE_TYPES
from args import Arguments
from astropy_healpix import HEALPix, boundaries_lonlat
from matplotlib.collections import PolyCollection

class MapVarCommand(Command):
    def run(self, raw_args):
        args = Arguments(raw_args)
        
        selected_vars = [v for v in VARIABLE_TYPES.keys() if hasattr(args, v) and getattr(args, v)]
        
        if not selected_vars:
            print("Error: Please provide at least one variable to map.")
            print(f"Available variables: {list(VARIABLE_TYPES.keys())}")
            return

        # Setup HEALPix resolution based on user resolution argument
        target_nside = 60.0 / args.res
        nside = max(1, 2 ** int(np.round(np.log2(max(1, target_nside)))))
        hp = HEALPix(nside=nside, order='ring', frame='icrs')

        # Layout configuration
        fig, axes = plt.subplots(1, len(selected_vars), figsize=(8 * len(selected_vars), 8) if args.polar else (12, 6), squeeze=False, subplot_kw={'projection': 'polar'} if args.polar else {})
        axes = axes.flatten()

        for ax, var_key in zip(axes, selected_vars):
            var = VARIABLE_TYPES[var_key]
            
            if args.polar:
                print(f"Extracting {var['name']} into HEALPix (nside={nside}, graining={args.graining})...")
                healpix_data = var['extractor']().extract_polar(
                    self, 180, 0, 180, args.range, hp, args.graining, args.sigma, args.normalize
                )
                
                all_pixels = np.arange(hp.npix)
                _, lat = hp.healpix_to_lonlat(all_pixels)
                dec = lat.deg
                
                # Filter pixels by hemisphere
                valid_pixels = all_pixels[dec >= 0] if args.north else all_pixels[dec <= 0]
                
                verts = []
                polygon_values = []
                
                for pix_id in valid_pixels:
                    val = healpix_data[pix_id]
                    if np.isnan(val):
                        continue
                        
                    # Get the exact curved boundaries (corners) of the HEALPix pixel
                    b_lon, b_lat = boundaries_lonlat(pix_id, nside=hp.nside, step=1)
                    b_ra = b_lon.deg
                    b_dec = b_lat.deg
                    
                    # Convert RA to radians and unwrap to prevent RA=0/360 wrapping glitches
                    theta = np.unwrap(np.radians(b_ra), period=2 * np.pi)
                    r = 90 - b_dec if args.north else b_dec + 90

                    # Ensure they are 1D arrays before stacking
                    theta = np.atleast_1d(theta).flatten()
                    r = np.atleast_1d(r).flatten()
                    
                    # Stack into (N, 2) polygon vertex array
                    poly_verts = np.column_stack((theta, r))
                    verts.append(poly_verts)
                    polygon_values.append(val)
                
                # 2. Render all HEALPix polygons efficiently using PolyCollection
                poly_collection = PolyCollection(verts, cmap='magma', edgecolors='none', alpha=0.9)
                poly_collection.set_array(np.array(polygon_values))
                im = ax.add_collection(poly_collection)
                
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_ylim(0, 90)
                
                ax.set_yticks([0, 30, 60, 90])
                if args.north:
                    ax.set_yticklabels(['90°', '60°', '30°', '0°'])
                    plt.suptitle('Mapped Variables (Northern Hemisphere - HEALPix Tessellation)', y=1.05)
                else:
                    ax.set_yticklabels(['-90°', '-60°', '-30°', '0°'])
                    plt.suptitle('Mapped Variables (Southern Hemisphere - HEALPix Tessellation)', y=1.05)
                    
                ax.set_xticks(np.radians(np.arange(0, 360, 45)))
                ax.set_xticklabels([f'{angle}°' for angle in np.arange(0, 360, 45)])
                ax.set_title(f"{var['name']} (nside={nside})", pad=20)
                
            else:
                # Cartesian fallback uses standard grid extraction
                print(f"Extracting {var['name']} ({args.graining})...")
                ra_min, ra_max = max(args.asc - args.area, 0), min(args.asc + args.area, 360)
                dec_min, dec_max = max(args.dec - args.area, -90), min(args.dec + args.area, 90)
                ra_edges = np.arange(ra_min, ra_max + args.res, args.res)
                dec_edges = np.arange(dec_min, dec_max + args.res, args.res)
                
                data = var['extractor']().extract_cartesian(
                    self, args.asc, args.dec, args.area, args.range, ra_edges, dec_edges, args.graining, args.normalize
                )
                im = ax.imshow(data.T, origin='lower', extent=[ra_min, ra_max, dec_min, dec_max], cmap='magma', aspect='auto')
                ax.set_xlabel('Right Ascension (deg)')
                ax.set_ylabel('Declination (deg)')
                ax.set_title(f"{var['name']} ({args.graining.title()})")
                plt.suptitle(f'Mapped Variables Centered at RA={args.asc}, Dec={args.dec}, Area={args.area}°', y=1.02)

            # Ensure colorbar scales correctly based on projection type
            if args.polar:
                fig.colorbar(poly_collection, ax=ax, label=var['name'], fraction=0.046, pad=0.08)
            else:
                fig.colorbar(im, ax=ax, label=var['name'], fraction=0.046, pad=0.04)
            
        plt.tight_layout()
        var_names = "_".join(selected_vars)
        
        if args.polar:
            hemisphere = "North" if args.north else "South"
            filename = f'figures/mapvar_{var_names}_{hemisphere}Hemisphere_HEALPix_Poly.png'
        else:
            filename = f'figures/mapvar_{var_names}_RA{args.asc}_Dec{args.dec}(graining={args.graining}).png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
        plt.show()