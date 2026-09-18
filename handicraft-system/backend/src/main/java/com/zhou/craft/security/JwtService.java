package com.zhou.craft.security;
import com.zhou.craft.model.UserAccount;import io.jsonwebtoken.*;import io.jsonwebtoken.security.Keys;import org.springframework.beans.factory.annotation.Value;import org.springframework.stereotype.Component;import javax.crypto.SecretKey;import java.nio.charset.StandardCharsets;import java.time.Duration;import java.util.Date;
@Component public class JwtService{
private final SecretKey key;private final long expirationMs;
public JwtService(@Value("${app.jwt.secret}")String secret,@Value("${app.jwt.expiration-hours}")long hours){key=Keys.hmacShaKeyFor(secret.getBytes(StandardCharsets.UTF_8));expirationMs=Duration.ofHours(hours).toMillis();}
public String create(UserAccount u){var now=new Date();return Jwts.builder().subject(u.getUsername()).claim("uid",u.getId()).claim("name",u.getDisplayName()).claim("role",u.getRole()).issuedAt(now).expiration(new Date(now.getTime()+expirationMs)).signWith(key).compact();}
public AuthContext.UserPrincipal parse(String token){var c=Jwts.parser().verifyWith(key).build().parseSignedClaims(token).getPayload();return new AuthContext.UserPrincipal(c.get("uid",Long.class),c.getSubject(),c.get("name",String.class),c.get("role",String.class));}
}