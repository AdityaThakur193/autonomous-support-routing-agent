import duckdb

print("Extracting @AmazonHelp threads using DuckDB...")

# Minimal, efficient query to grab Customer Inbound -> AmazonHelp Outbound pairs
# We join the dataset to itself: brand tweets responding to customer tweets.
query = """
SELECT 
    customer.text AS customer_message,
    brand.text AS brand_response
FROM read_csv_auto('data/twcs.csv', ALL_VARCHAR=TRUE) brand
JOIN read_csv_auto('data/twcs.csv', ALL_VARCHAR=TRUE) customer 
  ON brand.in_response_to_tweet_id = customer.tweet_id
WHERE brand.author_id = 'AmazonHelp'
  AND customer.inbound = 'True'
"""

# Execute and save directly to a new, small CSV
df = duckdb.query(query).df()
df.to_csv('data/amazon_pairs.csv', index=False)

print(f"Done! Saved {len(df)} conversation pairs to 'data/amazon_pairs.csv'.")
