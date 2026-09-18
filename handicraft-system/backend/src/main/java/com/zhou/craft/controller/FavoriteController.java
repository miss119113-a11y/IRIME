package com.zhou.craft.controller;
import com.zhou.craft.model.*;import com.zhou.craft.repo.*;import com.zhou.craft.security.AuthContext;import org.springframework.web.bind.annotation.*;import java.util.*;
@RestController @RequestMapping("/api/favorites")public class FavoriteController{
private final WorkFavoriteRepository fav;private final CraftWorkRepository works;public FavoriteController(WorkFavoriteRepository f,CraftWorkRepository w){fav=f;works=w;}
@GetMapping public List<CraftWork> list(){Long uid=AuthContext.require().id();return fav.findByUserIdOrderByCreatedAtDesc(uid).stream().map(x->works.findById(x.getWorkId()).orElse(null)).filter(Objects::nonNull).toList();}
@GetMapping("/work/{id}")public Map<String,Object> status(@PathVariable Long id){return Map.of("favorite",fav.findByUserIdAndWorkId(AuthContext.require().id(),id).isPresent());}
@PostMapping("/work/{id}")public Map<String,Object> toggle(@PathVariable Long id){Long uid=AuthContext.require().id();works.findById(id).orElseThrow();var c=fav.findByUserIdAndWorkId(uid,id);boolean state;if(c.isPresent()){fav.delete(c.get());state=false;}else{fav.save(new WorkFavorite(uid,id));state=true;}return Map.of("favorite",state);}
}