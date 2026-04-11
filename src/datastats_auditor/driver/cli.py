import argparse
from datastats_auditor import logger
from datastats_auditor.utils.utils import discover_plugins
from datastats_auditor.engine.orchestrator import compute_stats_and_drift
from datastats_auditor.stats.registry  import registry
import os
import yaml


_REQUIRED_CONFIG_SECTIONS = set(["stats", "drift", "datacard"])

discover_plugins()


def parse_args():
    parser = argparse.ArgumentParser(description="datastats_auditor CLI")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    return parser.parse_args()


def run_datastats_auditor_with_config():
    args = parse_args()
    config_path = args.config
    
    if not os.path.exists(config_path):
        msg = f"Configuration file not found: {config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    else:
        logger.info(f"Running datastats_auditor with config: {config_path}")
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded: {config}")
        except Exception as e:
            msg = f"Error reading configuration file: {e}"
            logger.error(msg)
            raise e

    config_keys = set(config.keys())
    if config_keys == _REQUIRED_CONFIG_SECTIONS:
        logger.info(f"All required config params are present.")
    else:
        missing_keys = _REQUIRED_CONFIG_SECTIONS - config_keys
        msg = f"Missing required config params: {missing_keys}. Required config params are: {_REQUIRED_CONFIG_SECTIONS}"
        logger.error(msg)
        raise KeyError(msg)
    logger.info(f"Successfully checked config params")
      
    logger.info("Start Reading config params")
    try:
        stats_config = config["stats"]["params"]
        split_stat_service_name = config["stats"]["name"]
        logger.info(f"split_stat_service_name: {split_stat_service_name}")
        split_stat_service_status = config["stats"]["status"]
        split_service = registry.get(name=split_stat_service_name,
                                     status=split_stat_service_status
                                     )
        services_pipeline = []
        orchestrator_params = {}
        if not isinstance(stats_config, list):
            stats_config = [stats_config]
        for conf in stats_config:
            service_name = conf["name"]
            logger.info(f"service_name: {service_name}")
            service_status = conf["status"]
            com_service = registry.get(name=service_name, status=service_status)
            if com_service is None:
                raise ValueError(f"service with name: {service_name} status: {service_status} is not present in the registry")
        
            logger.info(f"com_service: {com_service}")
            components = conf["components"]
            comp_service_params = conf.get("params", {}) or {}
            logger.info(f"comp_service_params: {comp_service_params}")
            component_instances = {}
                
            for comp, component in components.items():
                comp_name = component.pop("name")
                logger.info(f"comp_name: {comp_name}")
                comp_status = component.pop("status")
                component_cls = registry.get(name=comp_name, status=comp_status)
                logger.info(f"component_cls: {component_cls}")
                comp_params = component.get("params") or {}
                component_instances[comp] = component_cls 
                component_instances.update(comp_params)
                logger.info(f"comp_params: {comp_params}")
                comp_service_params.update(component_instances)
            services = com_service(**comp_service_params) 
            services_pipeline.append(services)
        
        card_config = config["datacard"]  
        card_creator_name = card_config["name"]
        card_creator_status = card_config["status"]
        card_params = card_config.get("params") or {}
        card_creater = registry.get(name=card_creator_name, status=card_creator_status)  
        
        
        orchestrator_params["card_creator"] = card_creater
        orchestrator_params.update(card_params)
        card_components = card_config["components"]  
        
        for _comp_nm, compsetup in card_components.items():
            datacard_comp_name = compsetup.get("name")
            datacard_comp_status = compsetup.get("status")
            datacard_comp_params = compsetup.get("params") or {}
            
            datacard_component_cls = registry.get(name=datacard_comp_name,
                                                  status=datacard_comp_status
                                                  )
            orchestrator_params[_comp_nm] = datacard_component_cls
            orchestrator_params.update(datacard_comp_params)
        
        drift_config = config.get("drift")
        if drift_config:
            drift_service_name = drift_config["name"]
            drift_service_status = drift_config["status"]
            drift_service_cls = registry.get(name=drift_service_name, 
                                            status=drift_service_status
                                            )
            orchestrator_params["drift_stats_service"] = drift_service_cls
            orchestrator_params.update(drift_config.get("params", {}))
            
            drift_comp = drift_config.get("components")
            if drift_comp:
                for _drift_comp_nm, _drift_comp_setup in drift_comp.items():
                    nm = _drift_comp_setup["name"]
                    _drift_status = _drift_comp_setup["status"]
                    _drift_comp_param = _drift_comp_setup.get("params") or {}
                    
                    drift_comp_cls = registry.get(name=nm, status=_drift_status)
                    
                    orchestrator_params[_drift_comp_nm] = drift_comp_cls
                    orchestrator_params.update(_drift_comp_param)
        logger.info(f"split_service: {split_service}")    
        logger.info(f"services_pipeline: {services_pipeline}")        
        split_service = split_service(services_cls=services_pipeline)       
        orchestrator_params["split_stats_service"] = split_service #services_pipeline
                    
        logger.info("Running the datastats and drift orchestrator")   
        compute_stats_and_drift(**orchestrator_params)  
        logger.info("Cpmpleted the datastats and drift orchestrator")                  
                    
    except KeyError as e:
        msg = f"Missing required config section or key: {e}"
        logger.error(msg)
        raise KeyError(msg)

if __name__ == "__main__":
    run_datastats_auditor_with_config()