import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = {}
    links = len(corpus[page])

    if links:            
        for link in corpus:
            distribution[link] = (1 - damping_factor) / len(corpus)

        for link in corpus[page]:
            distribution[link] += damping_factor / links
    else:
        for link in corpus:
            distribution[link] = 1 / len(corpus)
    
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_counts = {page: 0 for page in corpus}
    page = random.choice(list(corpus.keys()))

    for _ in range(n):
        page_counts[page] += 1
        model = transition_model(corpus, page, damping_factor)
        page = random.choices(list(model.keys()), weights=model.values(), k=1)[0]

    total = sum(page_counts.values())
    return {page: count / total for page, count in page_counts.items()}


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.
    """
    N = len(corpus)
    ranks = {page: 1 / N for page in corpus}
    threshold = 0.001

    while True:
        new_ranks = {}
        for page in corpus:
            total = 0
            for possible_page in corpus:
                links = corpus[possible_page]
                if links:
                    if page in links:
                        total += ranks[possible_page] / len(links)
                else:
                    # Treat pages with no links as linking to all pages
                    total += ranks[possible_page] / N
            new_ranks[page] = (1 - damping_factor) / N + damping_factor * total

        # Check for convergence
        if all(abs(new_ranks[page] - ranks[page]) < threshold for page in corpus):
            break
        ranks = new_ranks

    return ranks


if __name__ == "__main__":
    main()