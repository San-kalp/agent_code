from documents import hypothetical_question_documents

def main():
    for d in hypothetical_question_documents:
        print(d.page_content)
        print("Metadata :", d.metadata)
        print("-"*100)
    
main()