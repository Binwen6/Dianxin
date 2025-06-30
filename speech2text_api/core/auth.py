from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
from .config import settings

logger = logging.getLogger(__name__)

# 创建HTTP Bearer认证方案
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """
    验证API密钥
    
    Args:
        credentials: HTTP认证凭据
        
    Returns:
        验证通过的用户标识
        
    Raises:
        HTTPException: 当API密钥无效时
    """
    try:
        # 检查Authorization header格式
        if not credentials or not credentials.scheme.lower() == "bearer":
            raise HTTPException(
                status_code=401,
                detail="无效的认证格式。请使用 'Bearer {API_KEY}' 格式"
            )
        
        api_key = credentials.credentials
        
        # 验证API密钥
        if not api_key or api_key != settings.API_KEY:
            logger.warning(f"无效的API密钥尝试: {api_key[:10]}...")
            raise HTTPException(
                status_code=401,
                detail="无效的API密钥"
            )
        
        logger.info("API密钥验证成功")
        return api_key
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API密钥验证时发生错误: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="认证服务错误"
        ) 