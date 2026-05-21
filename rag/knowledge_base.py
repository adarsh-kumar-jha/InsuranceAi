"""
Insurance Policy Knowledge Base.
These chunks are retrieved via hybrid search and injected into Agent 2 as context.
"""

KNOWLEDGE_BASE = [
    {
        "id": "auto_001",
        "category": "auto",
        "title": "Auto Collision Coverage",
        "content": (
            "Collision coverage pays for damage to your vehicle when it collides with another "
            "vehicle or object, regardless of fault. This includes accidents at intersections, "
            "parking lot incidents, and single-car accidents. A deductible applies — typically "
            "$250 to $1,000. Collision coverage does NOT cover theft, weather damage, or "
            "hitting an animal."
        ),
        "keywords": ["collision", "car", "vehicle", "accident", "damage", "hit", "crash", "bumper", "dent"],
    },
    {
        "id": "auto_002",
        "category": "auto",
        "title": "Auto Comprehensive Coverage",
        "content": (
            "Comprehensive coverage protects your vehicle from non-collision events: theft, "
            "vandalism, natural disasters (hail, flood, fire), falling objects, and animal "
            "strikes. If your car is stolen and not recovered, comprehensive pays its actual "
            "cash value minus your deductible. Comprehensive is optional but required if your "
            "car is financed."
        ),
        "keywords": ["comprehensive", "theft", "stolen", "vandalism", "hail", "flood", "fire", "natural disaster"],
    },
    {
        "id": "auto_003",
        "category": "auto",
        "title": "Filing an Auto Claim",
        "content": (
            "To file an auto claim: (1) Ensure safety and call 911 if needed. (2) Document "
            "the scene with photos. (3) Exchange information with the other driver. (4) File "
            "a police report for accidents over $500 or involving injuries. (5) Contact your "
            "insurer within 24-72 hours. An adjuster will inspect the vehicle and estimate "
            "repair costs. You may use any licensed repair shop."
        ),
        "keywords": ["file claim", "report", "how to", "process", "steps", "accident", "adjuster", "repair"],
    },
    {
        "id": "auto_004",
        "category": "auto",
        "title": "Rental Car Coverage",
        "content": (
            "Rental reimbursement coverage pays for a rental car while your vehicle is being "
            "repaired after a covered claim. Typical limits are $30-$50/day up to $900-$1,500 "
            "per claim. This coverage must be added to your policy — it is not automatic. "
            "Without it, you may still be covered if the other driver was at fault via their "
            "liability coverage."
        ),
        "keywords": ["rental", "rental car", "loaner", "temporary car", "while repairs", "reimbursement"],
    },
    {
        "id": "auto_005",
        "category": "auto",
        "title": "Liability Coverage",
        "content": (
            "Auto liability coverage pays for bodily injury and property damage you cause to "
            "others in an accident. Bodily injury liability covers medical bills, lost wages, "
            "and legal fees. Property damage liability covers repairs to the other party's "
            "vehicle or property. Liability does NOT cover your own injuries or vehicle damage."
        ),
        "keywords": ["liability", "other driver", "injured someone", "damage their car", "fault", "at fault"],
    },
    {
        "id": "auto_006",
        "category": "auto",
        "title": "Uninsured / Underinsured Motorist Coverage",
        "content": (
            "Uninsured motorist (UM) coverage protects you when the at-fault driver has no "
            "insurance. Underinsured motorist (UIM) coverage applies when the at-fault driver's "
            "limits are insufficient. UM/UIM covers medical expenses, lost wages, and sometimes "
            "property damage depending on your state."
        ),
        "keywords": ["uninsured", "no insurance", "hit and run", "underinsured", "other driver no insurance"],
    },
    {
        "id": "home_001",
        "category": "home",
        "title": "Home Dwelling Coverage",
        "content": (
            "Dwelling coverage (Coverage A) protects the physical structure of your home — "
            "walls, roof, floors, windows, built-in appliances — from covered perils including "
            "fire, lightning, windstorm, hail, explosion, and vandalism. It covers the cost to "
            "rebuild your home, not the market value. Earthquakes and floods require separate policies."
        ),
        "keywords": ["house", "home", "dwelling", "structure", "roof", "walls", "rebuild", "fire", "wind"],
    },
    {
        "id": "home_002",
        "category": "home",
        "title": "Water Damage vs. Flood Coverage",
        "content": (
            "Standard homeowners insurance covers SUDDEN AND ACCIDENTAL water damage — burst "
            "pipes, appliance overflow, rain entering through storm damage. It does NOT cover "
            "flooding from external sources (rivers, storm surge, heavy rain). Flood damage "
            "requires a separate NFIP or private flood insurance policy. Gradual leaks and "
            "maintenance issues are also excluded."
        ),
        "keywords": ["water damage", "flood", "burst pipe", "leak", "flooding", "rain", "overflow", "NFIP"],
    },
    {
        "id": "home_003",
        "category": "home",
        "title": "Home Personal Property Coverage",
        "content": (
            "Personal property coverage (Coverage C) protects your belongings — furniture, "
            "electronics, clothing, appliances — against covered perils. Standard limits are "
            "50-70% of dwelling coverage. High-value items (jewelry, art, collectibles) may "
            "need scheduled endorsements. Coverage can be replacement cost value (RCV) or "
            "actual cash value (ACV) — RCV pays to replace without depreciation."
        ),
        "keywords": ["personal property", "belongings", "furniture", "electronics", "jewelry", "stolen", "replacement"],
    },
    {
        "id": "home_004",
        "category": "home",
        "title": "Filing a Home Insurance Claim",
        "content": (
            "To file a home claim: (1) Ensure everyone is safe. (2) Prevent further damage "
            "(cover broken windows, stop water leaks). (3) Document damage with photos and "
            "video. (4) Keep receipts for emergency expenses. (5) Contact your insurer within "
            "24-48 hours. An adjuster will inspect the damage. (6) Get contractor estimates. "
            "Do not permanently repair until the adjuster has inspected."
        ),
        "keywords": ["home claim", "file", "process", "steps", "adjuster", "inspection", "damage", "how to"],
    },
    {
        "id": "home_005",
        "category": "home",
        "title": "Fire Damage Claims",
        "content": (
            "Fire damage is typically covered under standard homeowners policies including "
            "accidental fires, wildfires, and lightning strikes. Coverage includes structural "
            "damage, personal property loss, and additional living expenses (ALE) if you must "
            "temporarily relocate. Arson committed by the policyholder is excluded. Document "
            "all damaged items for your claim."
        ),
        "keywords": ["fire", "smoke damage", "wildfire", "burn", "kitchen fire", "house fire", "arson"],
    },
    {
        "id": "home_006",
        "category": "home",
        "title": "Additional Living Expenses (ALE) Coverage",
        "content": (
            "Additional Living Expenses (ALE) / Loss of Use coverage pays for temporary "
            "housing, meals, and other costs if your home becomes uninhabitable due to a "
            "covered loss. This includes hotel stays, rental housing, and reasonable "
            "restaurant expenses above your normal food costs. ALE is typically 20-30% of "
            "dwelling coverage and applies until your home is repaired."
        ),
        "keywords": ["temporary housing", "hotel", "displaced", "living expenses", "uninhabitable", "ALE", "nowhere to live"],
    },
    {
        "id": "home_007",
        "category": "home",
        "title": "Burglary and Theft Coverage",
        "content": (
            "Homeowners insurance covers theft of personal property from your home, car, "
            "or while traveling. Police report is required. Standard limits apply — high-value "
            "items like jewelry, guns, or electronics may have sub-limits. Business property "
            "stolen from home is typically limited to $2,500 or may require a home business "
            "endorsement. Document all stolen items with serial numbers and receipts."
        ),
        "keywords": ["burglary", "theft", "stolen", "robbery", "break-in", "burglar", "missing items"],
    },
    {
        "id": "home_008",
        "category": "home",
        "title": "Storm and Wind Damage",
        "content": (
            "Standard homeowners policies cover damage from windstorms, hurricanes (in most "
            "states), hail, and lightning. Roof damage from wind or hail is a common claim. "
            "Some coastal states require a separate hurricane or wind deductible, which may "
            "be 1-5% of your dwelling coverage amount rather than a flat dollar amount. "
            "Flood damage accompanying storms requires separate flood coverage."
        ),
        "keywords": ["storm", "wind", "hurricane", "hail", "lightning", "tornado", "roof damage", "shingles"],
    },
    {
        "id": "coverage_001",
        "category": "coverage",
        "title": "Understanding Deductibles",
        "content": (
            "A deductible is the amount you pay out-of-pocket before insurance coverage "
            "applies. Higher deductibles mean lower premiums but more out-of-pocket costs "
            "when you claim. For a $500 deductible and $3,000 claim, you pay $500 and "
            "insurance pays $2,500. Some claims (liability) have no deductible. Comprehensive "
            "and collision deductibles are separate."
        ),
        "keywords": ["deductible", "out of pocket", "pay", "how much", "cost", "coverage amount"],
    },
    {
        "id": "coverage_002",
        "category": "coverage",
        "title": "Policy Limits",
        "content": (
            "Policy limits are the maximum amount your insurer will pay for a covered loss. "
            "Per-occurrence limits apply to each claim; aggregate limits apply across all "
            "claims in a policy period. For auto, limits are often shown as 100/300/100 — "
            "$100K per person / $300K per accident / $100K property damage. Review limits "
            "annually to ensure adequate coverage."
        ),
        "keywords": ["policy limit", "maximum", "coverage amount", "how much covered", "100/300", "limits"],
    },
    {
        "id": "coverage_003",
        "category": "coverage",
        "title": "Actual Cash Value vs Replacement Cost Value",
        "content": (
            "Actual Cash Value (ACV) pays what your property is worth today after depreciation. "
            "Replacement Cost Value (RCV) pays what it costs to replace the item new today. "
            "RCV policies cost more in premiums but pay significantly more for older items. "
            "Example: A 10-year-old TV worth $200 ACV but $500 RCV new. For total loss "
            "vehicles, ACV is typically used."
        ),
        "keywords": ["ACV", "RCV", "actual cash value", "replacement cost", "depreciation", "total loss"],
    },
    {
        "id": "coverage_004",
        "category": "coverage",
        "title": "Gap Insurance",
        "content": (
            "Gap insurance covers the difference between your vehicle's actual cash value "
            "and the amount you still owe on your auto loan or lease if the vehicle is totaled "
            "or stolen. Without gap insurance, you could owe thousands more than the insurance "
            "payout. Gap insurance is especially important for new vehicles, which depreciate "
            "quickly."
        ),
        "keywords": ["gap insurance", "total loss", "owe more than car worth", "underwater", "loan", "financed"],
    },
    {
        "id": "claims_001",
        "category": "claims",
        "title": "Claim Investigation Process",
        "content": (
            "After filing a claim, an adjuster investigates to verify coverage, assess damages, "
            "and determine liability. The investigation may include: reviewing police reports, "
            "inspecting damaged property, interviewing witnesses, reviewing medical records "
            "(for injury claims), and obtaining repair estimates. Simple claims may be resolved "
            "in days; complex claims can take weeks or months."
        ),
        "keywords": ["investigation", "adjuster", "process", "how long", "timeline", "review"],
    },
    {
        "id": "claims_002",
        "category": "claims",
        "title": "Claims Process Timeline",
        "content": (
            "Typical claim timelines: (1) Report claim — immediate. (2) Adjuster assigned — "
            "1-3 business days. (3) Initial inspection — 3-7 business days for property, "
            "varies for auto. (4) Settlement offer — within 15-30 days in most states. "
            "State regulations require insurers to acknowledge claims within a set timeframe "
            "(usually 10-15 days) and resolve them promptly."
        ),
        "keywords": ["timeline", "how long", "when", "days", "weeks", "settlement", "resolution"],
    },
    {
        "id": "claims_003",
        "category": "claims",
        "title": "Claim Denial and Appeals",
        "content": (
            "Claims may be denied for: policy exclusions, lapsed coverage, failure to report "
            "timely, fraud, or the loss not being covered. If denied, you have the right to "
            "appeal. Request a written explanation, review your policy, gather additional "
            "evidence, and file a written appeal. If unresolved, you can contact your state's "
            "insurance department or hire a public adjuster."
        ),
        "keywords": ["denied", "rejection", "appeal", "dispute", "unfair", "disagree"],
    },
    {
        "id": "claims_004",
        "category": "claims",
        "title": "Subrogation Rights",
        "content": (
            "Subrogation allows your insurer to recover costs from the at-fault party after "
            "paying your claim. For example, if another driver caused your accident and your "
            "insurer paid, they may sue the at-fault driver's insurer to recover those costs. "
            "You may need to cooperate in subrogation proceedings and avoid releasing the "
            "at-fault party from liability before notifying your insurer."
        ),
        "keywords": ["subrogation", "recovery", "at fault", "other party responsible", "third party"],
    },
    {
        "id": "compliance_001",
        "category": "compliance",
        "title": "Fair Claims Handling Requirements",
        "content": (
            "Insurers are legally required to handle claims fairly and promptly. This includes: "
            "acknowledging claims quickly, conducting thorough investigations, communicating "
            "claim status, not misrepresenting policy terms, not pressuring claimants to accept "
            "inadequate settlements, and paying undisputed amounts promptly. Violations may be "
            "reported to the state insurance department."
        ),
        "keywords": ["fair", "rights", "unfair practices", "regulations", "requirements", "laws"],
    },
    {
        "id": "compliance_002",
        "category": "compliance",
        "title": "Privacy and PII Protection",
        "content": (
            "Insurance companies are required to protect your personal information under state "
            "privacy laws and the Gramm-Leach-Bliley Act. Do not share sensitive identifiers "
            "(SSN, full policy numbers, bank account numbers) through unsecured channels. "
            "Use secure online portals or phone for sensitive information. Your insurer will "
            "never ask for full SSN or bank details through chat."
        ),
        "keywords": ["privacy", "personal information", "SSN", "sensitive", "security", "protect", "PII"],
    },
    {
        "id": "emergency_001",
        "category": "emergency",
        "title": "Emergency Claim Procedures",
        "content": (
            "In an emergency (active fire, flooding, structural collapse): (1) Evacuate and "
            "call 911 immediately. (2) Call your insurer's 24/7 emergency claims line. "
            "(3) Take emergency protective measures (board up windows, stop water). "
            "(4) Keep receipts for emergency expenses — these are typically covered. "
            "(5) Do not return to unsafe structures. Emergency claims are prioritized and "
            "expedited by most insurers."
        ),
        "keywords": ["emergency", "urgent", "immediate", "fire now", "flooding now", "right now", "active"],
    },
]
