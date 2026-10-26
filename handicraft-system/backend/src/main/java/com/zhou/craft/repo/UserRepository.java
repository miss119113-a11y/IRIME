package com.zhou.craft.repo;
import com.zhou.craft.model.UserAccount;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;
public interface UserRepository extends JpaRepository<UserAccount,Long>{
    Optional<UserAccount> findByUsernameAndPassword(String username,String password);
}
