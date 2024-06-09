from PIL import Image, ImageDraw, ImageFont

image_names = [
    "introduction_image.jpg",
    "why_this_study_matters.jpg",
    "gen_z_background.jpg",
    "financial_challenges.jpg",
    "homeownership_aspirations.jpg",
    "data_sources.jpg",
    "investment_strategies.jpg",
    "edward_thorp.jpg",
    "economic_variables.jpg",
    "behavioral_economics.jpg",
    "mpt.jpg",
    "home_costs.jpg",
    "methodology.jpg",
    "monte_carlo_simulations.jpg",
    "markov_chain_monte_carlo.jpg",
    "sensitivity_analysis.jpg",
    "gradient_descent.jpg",
    "linear_algebra.jpg",
    "risk_return_analysis.jpg",
    "lifecycle_investing.jpg",
    "simulation_outcomes.jpg",
    "economic_impact.jpg",
    "asset_allocation.jpg",
    "behavioral_insights.jpg",
    "down_payment.jpg",
    "home_cost_trends.jpg",
    "conclusion.jpg",
    "qa_image.jpg"
]

output_dir = "/Users/frank/Desktop/Project/Slides/Images/"

for image_name in image_names:
    img = Image.new('RGB', (800, 600), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    text = image_name.split('.')[0]
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (img.width - text_width) // 2
    text_y = (img.height - text_height) // 2
    d.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    img.save(output_dir + image_name)
