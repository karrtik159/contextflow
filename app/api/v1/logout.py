from fastapi import APIRouter, Request, Response, status

from app.api.deps import CurrentUser, DBSession
from app.core.security import blacklist_token

router = APIRouter(tags=["logout"])


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout(
    request: Request,
    response: Response,
    db: DBSession,
    current_user: CurrentUser,
):
    """Blacklist the current access + refresh tokens and clear the cookie."""
    # Blacklist the access token from the Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        access_token = auth_header.split(" ", 1)[1]
        await blacklist_token(access_token, db)

    # Blacklist the refresh token from the cookie
    refresh_token = request.cookies.get("refresh_token")
    if refresh_token:
        await blacklist_token(refresh_token, db)

    # Clear the refresh cookie
    response.delete_cookie(key="refresh_token", httponly=True, secure=True, samesite="lax")

    return {"detail": "Successfully logged out"}
