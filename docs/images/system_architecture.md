graph TD
    QT[QTrobot Hardware] --> SI[Sensor Input Processing]
    SI --> FE[Feature Extraction]
    FE --> CD[Confusion Detection]
    CD --> RP[Repair Policy Engine]
    RP --> RB[Robot Behavior]
    
    classDef hardware fill:#f39c12,stroke:#333,stroke-width:2px;
    classDef processing fill:#3498db,stroke:#333,stroke-width:2px;
    classDef detection fill:#e74c3c,stroke:#333,stroke-width:2px;
    classDef policy fill:#9b59b6,stroke:#333,stroke-width:2px;
    classDef behavior fill:#2ecc71,stroke:#333,stroke-width:2px;
    
    class QT hardware;
    class SI,FE processing;
    class CD detection;
    class RP policy;
    class RB behavior;
```

To generate the PNG image from this Mermaid diagram:

1. Install Mermaid CLI:
   ```bash
   npm install -g @mermaid-js/mermaid-cli
   ```

2. Generate the PNG:
   ```bash
   mmdc -i system_architecture.md -o system_architecture.png
   ```

Alternatively, you can use the Mermaid Live Editor (https://mermaid.live/) to generate the image manually.
