package com.zhou.craft.security;
import jakarta.servlet.http.*;import org.springframework.stereotype.Component;import org.springframework.web.servlet.HandlerInterceptor;import java.io.IOException;
@Component public class AuthInterceptor implements HandlerInterceptor{
private final JwtService jwt;public AuthInterceptor(JwtService jwt){this.jwt=jwt;}
public boolean preHandle(HttpServletRequest req,HttpServletResponse res,Object h)throws IOException{
String p=req.getRequestURI(),m=req.getMethod();if("OPTIONS".equals(m)||p.startsWith("/api/auth/")||("GET".equals(m)&&(p.startsWith("/api/works")||p.startsWith("/api/reviews/work/"))))return true;
String a=req.getHeader("Authorization");if(a==null||!a.startsWith("Bearer ")){reject(res,401,"请先登录");return false;}
try{var u=jwt.parse(a.substring(7));AuthContext.set(u);if(p.startsWith("/api/dashboard")&&!"ADMIN".equals(u.role())){reject(res,403,"无管理员权限");return false;}return true;}catch(Exception e){reject(res,401,"登录状态已过期，请重新登录");return false;}}
public void afterCompletion(HttpServletRequest r,HttpServletResponse s,Object h,Exception e){AuthContext.clear();}
private void reject(HttpServletResponse r,int s,String m)throws IOException{r.setStatus(s);r.setCharacterEncoding("UTF-8");r.setContentType("application/json;charset=UTF-8");r.getWriter().write("{\"success\":false,\"message\":\""+m+"\"}");}
}