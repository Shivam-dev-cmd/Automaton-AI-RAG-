import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from backend.app.config import settings

logger = logging.getLogger("document_store")

@dataclass
class DocumentChunk:
    id: str
    title: str
    category: str
    content: str
    url: Optional[str] = None
    caveats: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)

class DocumentStore:
    """
    Ingests and manages structured chunks from the Automaton AI knowledge base.
    Guarantees that all company facts, caveats, and guardrails are cleanly indexed.
    """
    def __init__(self, json_path: Optional[Path] = None):
        self.json_path = json_path or settings.KB_JSON_PATH
        self.chunks: List[DocumentChunk] = []
        self._category_map: Dict[str, List[DocumentChunk]] = {}
        self.load()

    def load(self):
        if not self.json_path.exists():
            logger.error(f"Knowledge base file not found at: {self.json_path}")
            return

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks: List[DocumentChunk] = []

        # 1. Company Overview
        company = data.get("company", {})
        diffs = company.get("differentiators_claimed", [])
        diff_str = "\n".join([f"- {d}" for d in diffs])
        content_company = (
            f"Company: {company.get('legal_name')}\n"
            f"Tagline: {company.get('tagline')}\n"
            f"Headquarters: {company.get('hq')}\n"
            f"Current Positioning: {company.get('positioning_current')}\n"
            f"Legacy Positioning: {company.get('positioning_legacy')}\n"
            f"Vision: {company.get('vision')}\n"
            f"Mission: {company.get('mission')}\n"
            f"Summary: {company.get('summary')}\n"
            f"Key Differentiators:\n{diff_str}"
        )
        chunks.append(DocumentChunk(
            id="company-overview",
            title="Automaton AI - Company Overview & Vision",
            category="company",
            content=content_company,
            url=data.get("meta", {}).get("website", "https://automatonai.com"),
            caveats="Metrics and superlatives are company claims. Current positioning is LLMOps/DLOps enterprise platform.",
            keywords=["automaton ai", "legal name", "tagline", "vision", "mission", "headquarters", "pune", "hinjewadi", "about", "who are you"],
            suggested_questions=["What is Automaton AI?", "Where is Automaton AI located?", "What are Automaton AI's core differentiators?"]
        ))

        # 2. Contact Information & Office Locations
        contact = data.get("contact", {})
        phones = ", ".join(contact.get("phones", []))
        form = contact.get("contact_form", {})
        form_topics = ", ".join(form.get("topics_connect_page", []))
        content_contact = (
            f"Email General: {contact.get('email_general')}\n"
            f"Email Careers: {contact.get('email_careers')}\n"
            f"Email Sales: {contact.get('email_sales')}\n"
            f"Phones: {phones}\n"
            f"Footer Address: {contact.get('address_footer')}\n"
            f"Connect Page Address: {contact.get('address_connect_page')}\n"
            f"Address Guidance: {contact.get('address_note')}\n"
            f"Connect Form URL: {form.get('url')}\n"
            f"Connect Form Topics: {form_topics}\n"
            f"Social Links: LinkedIn ({contact.get('social', {}).get('linkedin')}), YouTube ({contact.get('social', {}).get('youtube')}), Twitter ({contact.get('social', {}).get('x_twitter')})"
        )
        chunks.append(DocumentChunk(
            id="contact-info",
            title="Contact Information, Addresses & Connect Form",
            category="contact",
            content=content_contact,
            url=form.get("url", "https://automatonai.com/connect/"),
            caveats="The footer and connect page list slightly different addresses in Hinjewadi, Pune. Prefer the footer address (Suratwala Mark Plazzo).",
            keywords=["contact", "email", "phone", "address", "location", "connect", "reach out", "support", "office", "pune"],
            suggested_questions=["How can I contact Automaton AI?", "What is Automaton AI's office address?", "Where can I submit a demo request?"]
        ))

        # 3. People / Leadership
        people = data.get("people", [])
        people_lines = []
        for p in people:
            people_lines.append(f"- {p.get('name')} ({p.get('role')}): {p.get('bio')}")
        chunks.append(DocumentChunk(
            id="company-leadership",
            title="Automaton AI Leadership & Founders",
            category="leadership",
            content="Automaton AI Leadership Team:\n" + "\n".join(people_lines),
            url="https://automatonai.com/about-us/",
            caveats="Official leadership team consists of Bhushan Muthiyan (Founder), Mukesh Muthiyan (CTO), and Narendra Firodia (Investor/Adviser). Note third-party MCA records list directors Chandan & Girish Muthiyan.",
            keywords=["founder", "cto", "leadership", "ceo", "bhushan muthiyan", "mukesh muthiyan", "narendra firodia", "team", "investor"],
            suggested_questions=["Who founded Automaton AI?", "Who is the CTO of Automaton AI?", "Who are the key people behind Automaton AI?"]
        ))

        # 4. Products
        for prod in data.get("products", []):
            prod_id = prod.get("id")
            name = prod.get("name")
            category = prod.get("category")
            urls = prod.get("urls", [])
            primary_url = urls[0] if urls else "https://automatonai.com"
            one_liner = prod.get("one_liner", "")
            
            content_parts = [f"Product: {name} ({category})", f"Summary: {one_liner}"]
            
            if "lifecycle_stages" in prod:
                content_parts.append("Lifecycle Stages:")
                for st in prod["lifecycle_stages"]:
                    covers = ", ".join(st.get("covers", []))
                    content_parts.append(f"  * {st.get('stage')}: {covers}")
            
            if "capabilities" in prod:
                content_parts.append("Key Capabilities:\n" + "\n".join([f"  * {c}" for c in prod["capabilities"]]))
            
            if "frameworks" in prod:
                v2_fw = ", ".join(prod["frameworks"].get("v2_page", []))
                content_parts.append(f"Supported Frameworks: {v2_fw}")
                
            if "deployment_targets" in prod:
                targets = ", ".join(prod.get("deployment_targets", []))
                content_parts.append(f"Deployment Targets: {targets}")
                
            if "label_types" in prod:
                ltypes = ", ".join(prod.get("label_types", []))
                content_parts.append(f"Supported Annotation Types: {ltypes}")
                
            if "differentiators" in prod:
                diffs = ", ".join(prod.get("differentiators", []))
                content_parts.append(f"Key Differentiators: {diffs}")
                
            if "claims" in prod:
                claims = "; ".join(prod.get("claims", []))
                content_parts.append(f"Company Claims: {claims}")
                
            if "pricing" in prod:
                content_parts.append(f"Pricing: {prod.get('pricing')}")
                
            if "status" in prod:
                content_parts.append(f"Status / Note: {prod.get('status')}")

            caveat = None
            if prod_id == "advit-studio":
                caveat = "Pricing is not publicly listed. Must be routed to the Connect form. Features include on-premise, air-gapped, and DPDP compliance."
            elif prod_id == "docugpt":
                caveat = "Product page returned 404 at crawl time. Do not extrapolate features beyond enterprise doc extraction (99.2% accuracy claimed) and KYC/loan processing."
            elif prod_id == "tenderai":
                caveat = "Named in site footer only (returned 404). Features are unknown and strictly guarded from hallucination. Direct to sales/contact."

            keywords = [name.lower(), prod_id]
            if "advit" in prod_id:
                keywords.extend(["advit", "dlops", "llmops", "lora", "qlora", "rlhf", "air-gapped", "jetson", "onnx", "tensorrt", "dpdp", "labeling"])
            elif "adapt" in prod_id:
                keywords.extend(["automl", "tabular", "feature engineering", "predictive"])
            elif "docugpt" in prod_id:
                keywords.extend(["docugpt", "document ai", "rfp", "ocr", "kyc", "contracts"])
            elif "tenderai" in prod_id:
                keywords.extend(["tenderai", "tender ai", "government tenders"])
            elif "drone" in prod_id:
                keywords.extend(["drone", "aerial", "surveillance", "crop monitoring"])

            chunks.append(DocumentChunk(
                id=f"prod-{prod_id}",
                title=f"Product: {name}",
                category="product",
                content="\n".join(content_parts),
                url=primary_url,
                caveats=caveat,
                keywords=keywords,
                suggested_questions=[f"What is {name}?", f"What are the capabilities of {name}?", "How do I get access or pricing?"]
            ))

        # 5. Services
        for svc in data.get("services", []):
            svc_id = svc.get("id")
            name = svc.get("name")
            url = svc.get("url")
            summary = svc.get("summary", "")
            process = svc.get("process", [])
            content_svc = [f"Service: {name}", f"Summary: {summary}"]
            if process:
                content_svc.append("Process / Workflow:")
                content_svc.extend([f"  {idx+1}. {step}" for idx, step in enumerate(process)])
            if "annotation_types" in svc:
                content_svc.append(f"Annotation Types: {', '.join(svc.get('annotation_types', []))}")
            if "advantages" in svc:
                content_svc.append(f"Advantages: {', '.join(svc.get('advantages', []))}")

            chunks.append(DocumentChunk(
                id=f"service-{svc_id}",
                title=f"Service: {name}",
                category="service",
                content="\n".join(content_svc),
                url=url,
                caveats="Attribute workflow claims to Automaton AI.",
                keywords=[name.lower(), svc_id, "service", "consultancy", "data labeling", "custom ai"],
                suggested_questions=[f"Tell me about Automaton AI's {name}", "What is the process for custom AI development?"]
            ))

        # 6. Model Gallery
        models = data.get("model_gallery", [])
        m_lines = [f"- {m.get('name')}: {m.get('purpose')}" for m in models]
        chunks.append(DocumentChunk(
            id="model-gallery-catalog",
            title="Pre-built Model Gallery Catalog",
            category="models",
            content="Automaton AI Model Gallery (Ready-to-use Computer Vision Models):\n" + "\n".join(m_lines) + "\nNote: Demos available on request via Connect form.",
            url="https://automatonai.com/model-gallery/",
            caveats="Pre-trained models are image recognition/computer vision models.",
            keywords=["model gallery", "pre-built models", "face restoration", "dehazing", "pomegranate disease", "power lines", "car parts segmentation", "enlightening", "upscaling"],
            suggested_questions=["What models are in the Automaton AI Model Gallery?", "Does Automaton AI have a model for defect detection?"]
        ))

        # 7. Data Zoo (Datasets Marketplace)
        datasets = data.get("datasets", [])
        ds_lines = []
        for ds in datasets:
            classes = f" (Classes: {ds.get('classes')})" if ds.get('classes') else ""
            ds_lines.append(f"- {ds.get('name')}: {ds.get('type')}, Size: {ds.get('size_gb')} GB, Resolutions: {ds.get('resolutions')}, Labels: {ds.get('labels')}{classes}")
        formats = ", ".join(data.get("dataset_output_formats", []))
        chunks.append(DocumentChunk(
            id="data-zoo-catalog",
            title="Data Zoo - Annotated Datasets Marketplace",
            category="datasets",
            content=f"Data Zoo Datasets Catalog:\n" + "\n".join(ds_lines) + f"\nSupported Output Formats: {formats}\nQuote available on request via Data Zoo form.",
            url="https://automatonai.com/data-zoo/",
            caveats="Datasets are available for licensing; quotes provided upon inquiry.",
            keywords=["data zoo", "datasets", "annotated datasets", "corn leaf", "cotton", "vehicles", "pomegranate data", "restaurant data", "drone data", "yolo", "coco"],
            suggested_questions=["What datasets are available in Data Zoo?", "What output formats does Data Zoo support?"]
        ))

        # 8. Industries
        for ind in data.get("industries", []):
            name = ind.get("name")
            url = ind.get("url")
            summary = ind.get("summary", "")
            status = ind.get("page_status", "fetched")
            homepage_line = ind.get("homepage_line", "")
            use_cases = ind.get("use_cases", [])
            content_ind = [f"Industry Solution: {name}", f"Summary: {summary}"]
            if homepage_line:
                content_ind.append(f"Focus: {homepage_line}")
            if use_cases:
                content_ind.append("Use Cases:\n" + "\n".join([f"  * {u}" for u in use_cases]))
            
            caveat = None
            if status == "not_reached":
                caveat = f"Detailed page for {name} was not reached during site crawl. Information is limited to homepage summary. State limitations honestly."

            chunks.append(DocumentChunk(
                id=f"industry-{name.lower().replace(' ', '-').replace('&', 'and')}",
                title=f"Industry: {name}",
                category="industry",
                content="\n".join(content_ind),
                url=url,
                caveats=caveat,
                keywords=["industry", name.lower(), "solutions", "vertical"],
                suggested_questions=[f"How does Automaton AI serve the {name} industry?", f"What are Automaton AI's {name} use cases?"]
            ))

        # 9. Clients & Case Studies
        clients = data.get("clients_claimed", {})
        logos = ", ".join(clients.get("logos_shown", []))
        case_studies = clients.get("case_studies_v2_page", [])
        cs_lines = [f"- {cs.get('customer')} ({cs.get('sector')}): {cs.get('summary')}" for cs in case_studies]
        content_clients = (
            f"Clients & Enterprises (Company Claims):\n"
            f"Logos displayed: {logos}\n"
            f"Featured Case Studies:\n" + "\n".join(cs_lines) + "\n"
            f"Note on claims: Website claims vary between '40+ enterprises' (homepage meta description) and '50+ organisations / deployments' (homepage body and V2 page). These are company statements."
        )
        chunks.append(DocumentChunk(
            id="clients-case-studies",
            title="Clients, Partners & Case Studies (Apollo Tyres, ISRO, MSRDC)",
            category="case_studies",
            content=content_clients,
            url="https://automatonai.com/advit-studio-v2/",
            caveats="Customer relationships and metrics are stated by Automaton AI and not independently verified. Note 40+ vs 50+ discrepancy.",
            keywords=["clients", "customers", "apollo tyres", "isro", "satsure", "msrdc", "pune-mumbai expressway", "case studies", "who uses automaton ai"],
            suggested_questions=["Who are Automaton AI's notable clients?", "What did Automaton AI build for Apollo Tyres?", "What was the ISRO / SATSure project?"]
        ))

        # 10. Careers & Hiring
        careers = data.get("careers", {})
        openings = ", ".join(careers.get("openings_listed", []))
        genai_job = careers.get("job_description_detail", {}).get("GenAI Engineer", {})
        genai_duties = ", ".join(genai_job.get("duties", []))
        genai_reqs = ", ".join(genai_job.get("requirements", []))
        content_careers = (
            f"Careers at Automaton AI:\n"
            f"Apply URL: {careers.get('apply_url')}\n"
            f"Careers Email: {careers.get('email')}\n"
            f"Openings Listed: {openings}\n"
            f"Open to Unlisted Domains: Yes\n\n"
            f"Detailed Role Spotlight - GenAI Engineer:\n"
            f"Duties: {genai_duties}\n"
            f"Requirements: {genai_reqs}\n"
            f"Context: Works with foundational LLM companies supplying proprietary data for fine-tuning."
        )
        chunks.append(DocumentChunk(
            id="careers-hiring",
            title="Careers, Job Openings & Hiring at Automaton AI",
            category="careers",
            content=content_careers,
            url=careers.get("url", "https://automatonai.com/careers/"),
            caveats="Applicants should apply via https://automatonai.com/grow-with-us/ or send CV to careers@automatonai.com.",
            keywords=["careers", "jobs", "hiring", "genai engineer", "internship", "mlops engineer", "work at automaton ai", "apply"],
            suggested_questions=["What job openings are available at Automaton AI?", "How can I apply for a job at Automaton AI?", "What are the requirements for GenAI Engineer?"]
        ))

        # 11. Resources, Guides & Blogs
        resources = data.get("resources", {})
        blogs = resources.get("blogs", [])
        blog_lines = []
        for b in blogs:
            if "Restoration Test" in b.get("title", ""):
                continue
            blog_lines.append(f"- '{b.get('title')}' ({b.get('url')}): {b.get('summary')}")
        content_resources = "Enterprise AI Whitepapers & Guides:\n" + "\n".join(blog_lines)
        chunks.append(DocumentChunk(
            id="resources-blogs",
            title="Enterprise AI Guides: LLMOps & Agentic AI",
            category="resources",
            content=content_resources,
            url="https://automatonai.com/agentic-ai-indian-enterprises-guide/",
            caveats="Exclude placeholder test hero posts.",
            keywords=["agentic ai", "llmops", "guides", "blogs", "architecture", "deployment", "enterprise ai guide"],
            suggested_questions=["What is Automaton AI's perspective on Agentic AI?", "What are the 6 pillars of LLMOps according to Automaton AI?"]
        ))

        # 12. FAQ & Strict Guardrails (Pricing, Demos, Air-Gapped)
        faqs = data.get("faq_from_site_content", [])
        for idx, faq in enumerate(faqs):
            q = faq.get("q")
            a = faq.get("a")
            src = faq.get("source", "https://automatonai.com")
            
            # Special metadata for pricing
            caveat = None
            if "pricing" in q.lower():
                caveat = "Pricing is never public. Direct strictly to Connect form or info@automatonai.com. Never invent price numbers."
                
            chunks.append(DocumentChunk(
                id=f"faq-{idx+1}",
                title=f"FAQ: {q}",
                category="faq",
                content=f"Question: {q}\nAnswer: {a}\nSource: {src}",
                url=src,
                caveats=caveat,
                keywords=["faq", q.lower()],
                suggested_questions=[q]
            ))

        self.chunks = chunks
        for c in self.chunks:
            self._category_map.setdefault(c.category, []).append(c)

        logger.info(f"Loaded {len(self.chunks)} structured knowledge chunks across {len(self._category_map)} categories.")

    def get_all_chunks(self) -> List[DocumentChunk]:
        return self.chunks

    def get_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        return None

    def get_by_category(self, category: str) -> List[DocumentChunk]:
        return self._category_map.get(category, [])
