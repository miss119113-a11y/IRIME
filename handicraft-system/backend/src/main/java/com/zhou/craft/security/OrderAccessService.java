package com.zhou.craft.security;
import com.zhou.craft.model.CraftOrder;import org.springframework.http.HttpStatus;import org.springframework.stereotype.Component;import org.springframework.web.server.ResponseStatusException;import java.util.Arrays;
@Component public class OrderAccessService{
public AuthContext.UserPrincipal requireRole(String... roles){var p=AuthContext.require();if(Arrays.stream(roles).noneMatch(p.role()::equals))throw new ResponseStatusException(HttpStatus.FORBIDDEN,"当前角色无权执行此操作");return p;}
public void requireCustomer(CraftOrder o){var p=requireRole("USER","ADMIN");if("USER".equals(p.role())&&!p.id().equals(o.getCustomerId()))throw new ResponseStatusException(HttpStatus.FORBIDDEN,"不能操作其他用户订单");}
public void requireCreator(CraftOrder o){var p=requireRole("CREATOR","ADMIN");if("CREATOR".equals(p.role())&&!p.id().equals(o.getCreatorId()))throw new ResponseStatusException(HttpStatus.FORBIDDEN,"不能操作其他创作者订单");}
public void requireView(CraftOrder o){var p=AuthContext.require();boolean ok="ADMIN".equals(p.role())||("USER".equals(p.role())&&p.id().equals(o.getCustomerId()))||("CREATOR".equals(p.role())&&p.id().equals(o.getCreatorId()));if(!ok)throw new ResponseStatusException(HttpStatus.FORBIDDEN,"无权查看该订单");}
}