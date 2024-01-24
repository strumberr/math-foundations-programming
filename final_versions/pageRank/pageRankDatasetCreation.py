import csv
import random
import string

def generate_website_url():
    return ''.join(random.choices(string.ascii_lowercase, k=5)) + '.com'

def calculate_importance(num_links):
    return 1 / num_links if num_links > 0 else 0

def generate_dataset(num_websites):
    websites = []
    links = {}

    for i in range(num_websites):
        website_url = generate_website_url()
        num_links = random.randint(1, 10)
        websites.append((i, website_url, num_links))

        if random.random() < 0.10:
            print(f"Adding 100 links to {website_url}")
            websites[i] = (i, website_url, num_links + 20)
        elif random.random() < 0.2:
            print(f"Adding 20 links to {website_url}")
            websites[i] = (i, website_url, num_links + 10)


        links[website_url] = []


    for website_id, website_url, num_links in websites:
        possible_targets = [w for w in websites if w[1] != website_url]
        for _ in range(num_links):
            if possible_targets:
                target_website = random.choice(possible_targets)
                links[website_url].append(target_website[1])

    return websites, links

def write_to_csv(filename, websites, links):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'website_url', 'number_of_links', 'importance', 'links_it_contains'])

        for website in websites:
            website_id, website_url, num_links = website
            importance = calculate_importance(num_links)
            links_it_contains = ', '.join(links[website_url])
            writer.writerow([website_id, website_url, num_links, importance, links_it_contains])


num_websites = 500
websites, links = generate_dataset(num_websites)

csv_filename = 'pagerank_dataset.csv'
write_to_csv(csv_filename, websites, links)

print(f"Dataset generated and saved to {csv_filename}")
