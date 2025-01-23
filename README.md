# Deception: A Multi-Layered Security Framework for Software-Defined IoT Networks

This repository contains the source code, configuration files, and documentation for Deception, a multi-layered security framework designed for Software-Defined IoT (SD-IoT) networks. This framework focuses on DDoS detection, mitigation, and attack surface minimization.

## Project Overview

Deception combines data plane-based dynamic monitoring (DPDM) using P4, a multi-task ensemble for DDoS detection (MT-EDD) leveraging machine learning, and a deception technique to enhance security in SD-IoT environments. The project is structured to provide both a fully integrated implementation and independent modules for flexible exploration and deployment.

## Repository Structure

The repository is organized as follows:

*   **`DDM - Java - ONOS/`:** Contains Java source files related to the DDM module, which operates on the ONOS multi-controller.
*   **`DDM Module - Instructions.pdf`:** Provides instructions and explanations for the DDM module.
*   **`Deception- topology -json/`:** Contains JSON files defining the network topology for the deception technique.
*   **`Deception_Topo.py`:** Python script for implementing the deception topology.
*   **`DPDM Algorithm - Instructions.pdf`:** Provides instructions and explanations for the DPDM algorithm.
*   **`DPDM P4_App.p4`:** The P4 application for the DPDM module, implementing dynamic traffic monitoring in the data plane.
*   **`Feature Extractor - Instructions.pdf`:** Provides instructions and explanations for the feature extraction process.
*   **`Feature Extractor P4_App.p4`:** The P4 application for feature extraction in the data plane.
*   **`MT-EDD Algorithm - Instructions.pdf`:** Provides detailed instructions, explanations, and preprocessing steps for the MT-EDD algorithm.
*   **`MT-EDD Module Implementation.py`:** Python script implementing the MT-EDD algorithm for DDoS detection using machine learning.
*   **`P4_tutorial/`:** Contains tutorial materials and examples related to P4 programming.
*   **`P4-Mininet_App.py`:** Python script for setting up the SD-IoT network using Mininet-Wifi.
*   **`P4runtime_switch.py`:** Python script for interacting with the P4 switch using P4Runtime.
*   **`P4-SW[1-4]-runtime/`:** Contains JSON configuration files for the P4 switches.

## Getting Started

To run the full Deception framework, follow these general steps:

1.  **Environment Setup:** Ensure you have the necessary software installed:
    *   Mininet-Wifi
    *   P4 compiler (`p4c`)
    *   P4 runtime environment (e.g., `simple_switch`)
    *   POX and ONOS controllers
    *   Python 3 with required libraries (see `MT-EDD Module Implementation.py` for dependencies).
    *   Java Development Kit (JDK) for ONOS interaction.

2.  **Network Setup:** Use `P4-Mininet_App.py` and `Deception_Topo.py` to create the SD-IoT network topology in Mininet-Wifi.

3.  **P4 Switch Configuration:** Compile the P4 programs (`DPDM P4_App.p4`, `Feature Extractor P4_App.p4`) and configure the P4 switches using the provided JSON configuration files (`P4-SW[1-4]-runtime/`).

4.  **MT-EDD Training:** Run `MT-EDD Module Implementation.py` to train the machine learning models. Ensure that you have the CICIoT2023 dataset available and update the data loading path in the script.

5.  **Cloud Deployment (MT-EDD Prediction):** Deploy the trained MT-EDD models on a cloud server using the instructions provided in `MT-EDD Algorithm - Instructions.pdf` and the `predict_server.py` script.

6.  **DDM Setup:** Set up the ONOS controller and deploy the DDM module using the instructions in `DDM Module - Instructions.pdf`.

7.  **Integration:** Configure the communication between the P4 switches, the MT-EDD prediction server, and the ONOS controller to enable the complete framework functionality.

## Module-Specific Instructions

For detailed instructions on each module, please refer to the corresponding PDF files:

*   `DDM Module - Instructions.pdf`
*   `DPDM Algorithm - Instructions.pdf`
*   `Feature Extractor - Instructions.pdf`
*   `MT-EDD Algorithm - Instructions.pdf`

## Dependencies

The project has the following dependencies:

*   Mininet-Wifi
*   P4 compiler (`p4c`)
*   P4 runtime environment (e.g., `simple_switch`)
*   ONOS controller
*   Python 3
*   Java Development Kit (JDK)
*   Python libraries (see `MT-EDD Module Implementation.py` for a detailed list)
