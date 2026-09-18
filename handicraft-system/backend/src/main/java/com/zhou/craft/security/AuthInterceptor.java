package com.zhou.craft.security;

import jakarta.servlet.http.*;
import org.springframework.stereotype.Component;
import org.springframework.web.servlet.HandlerInterceptor;
import java.io.IOException;

@Component
public class AuthInterceptor implements HandlerInterceptor {
    private final JwtService jwt;
    public AuthInterceptor(JwtService jwt){this.jwt=jwt;}

    @Override
    public boolean preHandle(HttpServletRequest req,HttpServletResponse res,Object handler) throws IOException {
        String path=req.getRequestURI(),method=req.getMethod();
        boolean publicAuth="/api/auth/login".equals(path)||"/api/auth/register".equals(path);
        boolean publicRead="GET".equalsIgnoreCase(method)&&(path.startsWith("/api/works")||path.startsWith("/api/reviews/work/"));
        if("OPTIONS".equalsIgnoreCase(method)||publicAuth||publicRead) return true;

        String header=req.getHeader("Authorization");
        if(header==null||!header.startsWith("Bearer ")){reject(res,401,"请先登录");return false;}
        try{
            var user=jwt.parse(header.substring(7));
            AuthContext.set(user);
            if(path.startsWith("/api/dashboard")&&!"ADMIN".equals(user.role())){reject(res,403,"无管理员权限");return false;}
            return true;
        }catch(Exception e){
            AuthContext.clear();
            reject(res,401,"登录状态已过期，请重新登录");
            return false;
        }
    }

    @Override
    public void afterCompletion(HttpServletRequest request,HttpServletResponse response,Object handler,Exception ex){
        AuthContext.clear();
    }

    private void reject(HttpServletResponse res,int status,String msg) throws IOException{
        res.setStatus(status);
        res.setCharacterEncoding("UTF-8");
        res.setContentType("application/json;charset=UTF-8");
        res.getWriter().write("{\"success\":false,\"message\":\""+msg+"\"}");
    }
}
